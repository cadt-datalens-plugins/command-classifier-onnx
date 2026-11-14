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
export function encode(text, tokenizerData, maxLength) {
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
