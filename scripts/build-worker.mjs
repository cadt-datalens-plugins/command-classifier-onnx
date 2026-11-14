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
var session = null;
var tokenizerData = null;
var head = null;
var ready = false;

self.onmessage = async function (e) {
    var msg = e.data;
    if (msg.type === 'init') {
        try {
            importScripts(msg.ortUrl);
            ort.env.wasm.wasmBinary = new Uint8Array(msg.wasmBuffer);
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
            var enc = encode(msg.command, tokenizerData, 64);
            var len = enc.inputIds.length;
            var feeds = {
                input_ids: new ort.Tensor('int64', BigInt64Array.from(enc.inputIds.map(BigInt)), [1, len]),
                attention_mask: new ort.Tensor('int64', BigInt64Array.from(enc.attentionMask.map(BigInt)), [1, len]),
            };
            if (session.inputNames.indexOf('token_type_ids') >= 0) {
                feeds.token_type_ids = new ort.Tensor('int64', BigInt64Array.from(enc.tokenTypeIds.map(BigInt)), [1, len]);
            }
            var output = await session.run(feeds);
            var hidden = Object.values(output)[0].data;
            var emb = meanPool(Float32Array.from(hidden), enc.attentionMask, head.dim || 384);
            var score = headScore(emb, head);
            postMessage({type: 'result', id: msg.id, score: score, ms: Date.now() - t0, tokens: len});
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
