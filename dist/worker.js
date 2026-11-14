// dist/worker.js：打包产物（scripts/build-worker.mjs 生成，勿手改）——tokenizer + 池化/头 + ort-web 胶水
// src/tokenizer.js：BERT WordPiece 分词器（零依赖，Node 与浏览器同一实现）。
// 对齐 HuggingFace BertNormalizer（clean + 去重音符 + 小写 + CJK 切分）→ BertPreTokenizer
// （空白 + 标点切分）→ WordPiece（## 前缀贪心最长匹配）→ [CLS] … [SEP] 模板。
// tokenizerData 即 HF tokenizer.json（vocab / unk / continuing_subword_prefix）。

const CONTROL_RE = /[\0-\x08\x0B\x0C\x0E-\x1F]/g;
const WHITESPACE_RE = /\s+/g;
// HF BertTokenizer 的标点集：ASCII 标点 + Unicode 标点类别（P*/S* 简化按 P 区段即可覆盖命令文本）
const PUNCT_RE = /[\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E\u2000-\u206F\u3000-\u303F\uFF00-\uFFEF]/;
const CJK_RE = /[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF]/;

function stripAccents(text) {
    // NFD 分解后丢弃组合变音符号（Mn 类）
    const normalized = text.normalize('NFD');
    let out = '';
    for (const ch of normalized) {
        const code = ch.codePointAt(0);
        if (code >= 0x0300 && code <= 0x036F) continue; // combining diacritical marks
        out += ch;
    }
    return out;
}

function bertNormalize(text) {
    // NFKC 兼容分解先行：全角字符折叠为半角（红队 P1：全角化危险命令全部变 [UNK] 绕过）
    let out = String(text == null ? '' : text).normalize('NFKC')
        .replace(CONTROL_RE, '')
        .replace(WHITESPACE_RE, ' ')
        .trim();
    // CJK 字符前后补空格（HF chinese_nlp 处理）
    let spaced = '';
    for (const ch of out) {
        if (CJK_RE.test(ch)) spaced += ' ' + ch + ' ';
        else spaced += ch;
    }
    out = spaced.replace(WHITESPACE_RE, ' ').trim();
    out = stripAccents(out).toLowerCase();
    return out;
}

function isPunct(ch) {
    return PUNCT_RE.test(ch);
}

// BertPreTokenizer：先按空白切，再把标点逐字符拆出
function preTokenize(text) {
    const tokens = [];
    for (const chunk of bertNormalize(text).split(' ')) {
        if (!chunk) continue;
        let current = '';
        for (const ch of chunk) {
            if (isPunct(ch) || CJK_RE.test(ch)) {
                if (current) tokens.push(current);
                tokens.push(ch);
                current = '';
            } else {
                current += ch;
            }
        }
        if (current) tokens.push(current);
    }
    return tokens;
}

// WordPiece 贪心最长匹配；超长词 / 无法匹配 → [UNK]
function wordPiece(word, vocab, unkId, maxChars) {
    const limit = maxChars || 100;
    if (word.length > limit) return [unkId];
    const ids = [];
    let start = 0;
    while (start < word.length) {
        let end = word.length;
        let matchId = -1;
        while (start < end) {
            const piece = (start === 0 ? '' : '##') + word.slice(start, end);
            const id = vocab[piece];
            if (id !== undefined) {
                matchId = id;
                break;
            }
            end -= 1;
        }
        if (matchId === -1) return [unkId];
        ids.push(matchId);
        start = end;
    }
    return ids;
}

// encode：返回 {inputIds, attentionMask, tokenTypeIds}（定长 maxLength，截断保 [SEP]）
function encode(text, tokenizerData, maxLength) {
    const max = Number(maxLength) > 0 ? Number(maxLength) : 64;
    const vocab = tokenizerData.model.vocab;
    const unkId = vocab[tokenizerData.model.unk_token] || vocab['[UNK]'] || 0;
    const clsId = vocab['[CLS]'];
    const sepId = vocab['[SEP]'];
    const padId = vocab['[PAD]'] || 0;
    const budget = max - 2; // [CLS]/[SEP] 占位
    const pieces = [];
    for (const token of preTokenize(text)) {
        const ids = wordPiece(token, vocab, unkId, tokenizerData.model.max_input_chars_per_word);
        for (const id of ids) {
            if (pieces.length >= budget) break;
            pieces.push(id);
        }
        if (pieces.length >= budget) break;
    }
    const inputIds = [clsId, ...pieces, sepId];
    while (inputIds.length < max) inputIds.push(padId);
    const attentionMask = inputIds.map((id) => (id === padId ? 0 : 1));
    // BERT 单句：token_type 全 0
    const tokenTypeIds = inputIds.map(() => 0);
    return {inputIds, attentionMask, tokenTypeIds};
}

// src/engine.js：MiniLM 句向量池化 + 逻辑回归头打分（Node 训练与浏览器推理同一实现）。
// sentence-transformers 约定：对 last_hidden_state 做 attention-mask 加权平均池化 → 384 维句向量。

// mean pooling：非 padding token 的隐藏态均值
function meanPool(lastHiddenState, attentionMask, hiddenSize) {
    const size = hiddenSize || 384;
    const out = new Float32Array(size);
    let count = 0;
    for (let t = 0; t < attentionMask.length; t += 1) {
        if (!attentionMask[t]) continue;
        const offset = t * size;
        for (let d = 0; d < size; d += 1) out[d] += lastHiddenState[offset + d];
        count += 1;
    }
    if (count > 0) {
        for (let d = 0; d < size; d += 1) out[d] /= count;
    }
    return out;
}

// 逻辑回归头：σ(w·x + b)
function headScore(embedding, head) {
    let z = head.bias || 0;
    const weights = head.weights;
    for (let d = 0; d < weights.length; d += 1) z += weights[d] * embedding[d];
    return 1 / (1 + Math.exp(-z));
}

// 单样本 SGD 步（训练用，Node 侧）：返回新 bias
function headSgdStep(embedding, head, y, lr, l2) {
    const p = headScore(embedding, head);
    const g = p - y;
    for (let d = 0; d < head.weights.length; d += 1) {
        head.weights[d] -= lr * (g * embedding[d] + l2 * head.weights[d]);
    }
    return p;
}


// —— Worker 胶水（打包生成，勿手改）：ort-web 会话 + 消息协议 ——
var ortUrl = '';
var wasmBytes = null;
var session = null;
var tokenizerData = null;
var head = null;
var ready = false;

// fetch 垫片：blob Worker 里 importScripts(ort) 后，ort 按相对名 fetch ort-wasm-simd.wasm
// 会因 blob 作用域无 base URL 而失败；垫片把它定向到主线程直传的 wasm 字节（版本无关、完全离线）。
var realFetch = self.fetch.bind(self);
self.fetch = function (input, init) {
    if (wasmBytes && typeof input === 'string' && /ort-wasm[^/]*\.wasm(\?|$)/.test(input)) {
        return Promise.resolve(new Response(wasmBytes, {headers: {'Content-Type': 'application/wasm'}}));
    }
    return realFetch(input, init);
};

// 命令替换（$() / 反引号 / 进程替换 <()）内的命令文本提取（平衡扫描，支持嵌套；
// 引号内的括号按引号串跳过）。刻意不写正则、反引号用 fromCharCode——本文件是
// 模板字符串，字面的反斜杠与反引号都要双写转义，实测踩过一次坑（见下方分段正则注释）。
function subCommands(text) {
    var BT = String.fromCharCode(96);
    var out = [];
    var n = text.length;
    var i = 0;
    while (i < n) {
        var ch = text[i];
        if ((ch === '$' || ch === '<') && text[i + 1] === '(') {
            var depth = 1;
            var j = i + 2;
            while (j < n && depth > 0) {
                var c = text[j];
                if (c === '(') depth += 1;
                else if (c === ')') depth -= 1;
                else if (c === '\'' || c === '"') {
                    var quote = c;
                    j += 1;
                    while (j < n && text[j] !== quote) j += 1;
                }
                j += 1;
            }
            if (depth === 0) {
                var inner = text.slice(i + 2, j - 1);
                if (inner.trim()) {
                    var nested = subCommands(inner);
                    for (var k = 0; k < nested.length; k += 1) out.push(nested[k]);
                    out.push(inner);
                }
                i = j;
                continue;
            }
            i += 2;
            continue;
        }
        if (ch === BT) {
            var close = text.indexOf(BT, i + 1);
            if (close > i) {
                var inner2 = text.slice(i + 1, close);
                if (inner2.trim()) {
                    var nested2 = subCommands(inner2);
                    for (var k2 = 0; k2 < nested2.length; k2 += 1) out.push(nested2[k2]);
                    out.push(inner2);
                }
                i = close + 1;
                continue;
            }
        }
        i += 1;
    }
    return out;
}

self.onmessage = async function (e) {
    var msg = e.data;
    if (msg.type === 'init') {
        try {
            wasmBytes = new Uint8Array(msg.wasmBuffer);
            importScripts(msg.ortUrl);
            ort.env.wasm.numThreads = 1;
            ort.env.wasm.simd = true;
            session = await ort.InferenceSession.create(new Uint8Array(msg.modelBuffer), {executionProviders: ['wasm']});
            tokenizerData = msg.tokenizerData;
            head = msg.head;
            ready = true;
            postMessage({type: 'ready', inputs: session.inputNames, outputs: session.outputNames});
        } catch (err) {
            postMessage({type: 'error', message: 'init: ' + String((err && err.message) || err)});
        }
        return;
    }
    if (msg.type === 'score') {
        if (!ready) {
            postMessage({type: 'error', id: msg.id, message: 'worker not ready'});
            return;
        }
        try {
            var t0 = Date.now();
            // 红队 P0：多命令链的 benign 前缀会稀释危险段（meanPool 均值池化）——
            // 按 shell 命令分隔符切段、逐段推理取 max；与训练口径自洽（危险段命中正权）。
            // 正则在模板字符串里必须**双写反斜杠**（build 事故实录：单写时 \r \n \| 被
            // cooked 成真实控制符与裸竖线，产物正则带空分支逐字符切段——真机全命令
            // 0.87-0.99 静默失效，v0.2.1 及之前从未真正生效）。
            var segs = String(msg.command || '').split(/\r?\n|&&|\|\||;|\|/).map(function (x) { return x.trim(); }).filter(Boolean);
            // 红队 F1（2026-11-15）：命令替换内的命令与洪泛同段会被均值池化稀释
            // （实测最坏 0.1579 贴 0.15 阈值）——内层命令切成独立段参与 max（与主仓 lcSegment 同口径）
            var subs = [];
            for (var gi = 0; gi < segs.length; gi += 1) {
                var found = subCommands(segs[gi]);
                for (var fi = 0; fi < found.length; fi += 1) {
                    var subText = found[fi].trim();
                    if (subText) subs.push(subText);
                }
            }
            segs = segs.concat(subs);
            if (!segs.length) segs = [String(msg.command || '')];
            var best = 0, seen = 0;
            var truncated = false;
            for (var si = 0; si < segs.length && si < 48; si += 1) {
                var enc = encode(segs[si], tokenizerData, 64);
                var realLen = enc.attentionMask.reduce(function (a, b) { return a + b; }, 0);
                // 红队 P0：单段超过 62 token（截断丢尾）→ fail-closed 强制转人工
                if (realLen > 62) { truncated = true; best = 1; break; }
                var feeds = {
                    input_ids: new ort.Tensor('int64', BigInt64Array.from(enc.inputIds.map(BigInt)), [1, enc.inputIds.length]),
                    attention_mask: new ort.Tensor('int64', BigInt64Array.from(enc.attentionMask.map(BigInt)), [1, enc.inputIds.length]),
                };
                if (session.inputNames.indexOf('token_type_ids') >= 0) {
                    feeds.token_type_ids = new ort.Tensor('int64', BigInt64Array.from(enc.tokenTypeIds.map(BigInt)), [1, enc.inputIds.length]);
                }
                var output = await session.run(feeds);
                var hidden = Object.values(output)[0].data;
                var emb = meanPool(Float32Array.from(hidden), enc.attentionMask, head.dim || 384);
                var score = headScore(emb, head);
                if (score > best) best = score;
                seen += 1;
            }
            postMessage({type: 'result', id: msg.id, score: best, ms: Date.now() - t0, tokens: seen, truncated: truncated, segs: segs.length});
        } catch (err) {
            postMessage({type: 'error', id: msg.id, message: 'score: ' + String((err && err.message) || err)});
        }
        return;
    }
    if (msg.type === 'status') {
        postMessage({type: 'status', ready: ready, inputs: session ? session.inputNames : null});
    }
};
