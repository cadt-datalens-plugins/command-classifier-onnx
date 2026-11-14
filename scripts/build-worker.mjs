// scripts/build-worker.mjs：把 src/tokenizer.js + src/engine.js 与 Worker 胶水拼成 dist/worker.js
// （classic Worker 无法 import ESM，打包成单文件；Node 验证直接 import src/*.js 同源实现）。
// 运行：npm run build
import {readFileSync, writeFileSync, mkdirSync} from 'node:fs';
import {join, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..');

function stripExports(source) {
    return source.replace(/^export (function|const|let|var)/gm, '$1');
}

const tokenizer = stripExports(readFileSync(join(REPO, 'src/tokenizer.js'), 'utf8'));
const engine = stripExports(readFileSync(join(REPO, 'src/engine.js'), 'utf8'));

const glue = `
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
    if (wasmBytes && typeof input === 'string' && /ort-wasm[^/]*\\.wasm(\\?|$)/.test(input)) {
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
                else if (c === '\\'' || c === '"') {
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
            // 正则在模板字符串里必须**双写反斜杠**（build 事故实录：单写时 \\r \\n \\| 被
            // cooked 成真实控制符与裸竖线，产物正则带空分支逐字符切段——真机全命令
            // 0.87-0.99 静默失效，v0.2.1 及之前从未真正生效）。
            var segs = String(msg.command || '').split(/\\r?\\n|&&|\\|\\||;|\\|/).map(function (x) { return x.trim(); }).filter(Boolean);
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
`;

mkdirSync(join(REPO, 'dist'), {recursive: true});
const out = [
    '// dist/worker.js：打包产物（scripts/build-worker.mjs 生成，勿手改）——tokenizer + 池化/头 + ort-web 胶水',
    tokenizer,
    engine,
    glue,
].join('\n');
writeFileSync(join(REPO, 'dist/worker.js'), out);
console.log('已写出 dist/worker.js（' + Buffer.byteLength(out) + ' 字节）');
