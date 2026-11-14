// scripts/train-head.mjs：在 Node 侧用 onnxruntime-node 嵌入语料 → 训练逻辑回归头 → model/head.json。
// 运行前提：主仓库与本体仓库同级检出（跨仓库 import 语料生成器与金丝雀）。
// 运行：npm install && npm run train
import {writeFileSync, readFileSync} from 'node:fs';
import {join, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {encode} from '../src/tokenizer.js';
import {meanPool, headSgdStep, headScore} from '../src/engine.js';

const require = createRequire(import.meta.url);
const ort = require('onnxruntime-node');

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..');
const MAIN_REPO = join(REPO, '..', 'CaDt-DataLens');

const {generateCorpus, CANARIES} = await import('file://' + join(MAIN_REPO, 'scripts', 'classifier-corpus.mjs').replace(/\\/g, '/'));
const {LOCAL_CLASSIFIER_THRESHOLDS} = await import('file://' + join(MAIN_REPO, 'frontend/src/ai/ai-permissions.js').replace(/\\/g, '/'));

const MAX_LEN = 64;
const EPOCHS = 150;
const LR0 = 0.2;
const DECAY = 0.955;
const L2 = 1e-4;

const tokenizerData = JSON.parse(readFileSync(join(REPO, 'model/tokenizer.json'), 'utf8'));
const session = await ort.InferenceSession.create(join(REPO, 'model/model_quantized.onnx'));

// 单条命令 → 384 维句向量（一次性建小批队列，串行跑 930+576 条也只需几十秒）
async function embed(commands) {
    const vectors = [];
    const inames = session.inputNames;
    for (let i = 0; i < commands.length; i += 1) {
        const {inputIds, attentionMask, tokenTypeIds} = encode(commands[i], tokenizerData, MAX_LEN);
        const feeds = {
            input_ids: new ort.Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, MAX_LEN]),
            attention_mask: new ort.Tensor('int64', BigInt64Array.from(attentionMask.map(BigInt)), [1, MAX_LEN]),
        };
        if (inames.includes('token_type_ids')) {
            feeds.token_type_ids = new ort.Tensor('int64', BigInt64Array.from(tokenTypeIds.map(BigInt)), [1, MAX_LEN]);
        }
        const output = await session.run(feeds);
        const hidden = Object.values(output)[0].data; // last_hidden_state [1, L, 384]
        vectors.push(meanPool(Float32Array.from(hidden), attentionMask, 384));
    }
    return vectors;
}

async function main() {
    console.log('模型 IO：inputs=' + session.inputNames.join(',') + ' outputs=' + session.outputNames.join(','));
    const train = generateCorpus(20261114, {safe: 700, dangerous: 700, gray: 220});
    const valRaw = generateCorpus(8675309, {safe: 700, dangerous: 700, gray: 220});
    // 跨集合去重（与线性模型训练同一规则）
    const trainKeys = new Set(train.samples.map((s) => s.command.toLowerCase()));
    const val = {samples: valRaw.samples.filter((s) => !trainKeys.has(s.command.toLowerCase()))};

    console.log('嵌入训练集 ' + train.samples.length + ' 条…');
    const trainVectors = await embed(train.samples.map((s) => s.command));
    console.log('嵌入验证集 ' + val.samples.length + ' 条…');
    const valVectors = await embed(val.samples.map((s) => s.command));

    // 逻辑回归头：SGD + L2（与工厂线性模型同超参风格）
    const head = {weights: new Array(384).fill(0), bias: 0};
    // 灰色类（install/push/kill 等改环境）在嵌入空间语义上偏良性，过采样 ×3 保住边界
    // 过采样：灰色（语义偏良性的改环境类）×3、危险（引用路径碎片化等弱证据族）×2——
    // 冻结嵌入上逻辑回归的「策略语义」必须靠样本量压过「字面语义」
    const trainIdx = [];
    train.samples.forEach((s, i) => {
        trainIdx.push(i);
        if (s.cls === 'gray') trainIdx.push(i, i, i);
        else if (s.cls === 'dangerous') trainIdx.push(i, i);
    });
    let lr = LR0;
    for (let epoch = 0; epoch < EPOCHS; epoch += 1) {
        // 确定性洗牌
        for (let i = trainIdx.length - 1; i > 0; i -= 1) {
            const j = (epoch * 7919 + i * 31 + 17) % (i + 1);
            [trainIdx[i], trainIdx[j]] = [trainIdx[j], trainIdx[i]];
        }
        for (const idx of trainIdx) {
            headSgdStep(trainVectors[idx], head, train.samples[idx].label, lr, L2);
        }
        lr *= DECAY;
    }

    // 验证集指标：三档 allow 率 + 硬危险漏放
    const rows = val.samples.map((s, i) => ({command: s.command, cls: s.cls, y: s.label, score: headScore(valVectors[i], head)}));
    const summary = {};
    for (const cls of ['safe', 'dangerous', 'gray']) {
        const list = rows.filter((r) => r.cls === cls);
        const sorted = list.map((r) => r.score).sort((a, b) => a - b);
        summary[cls] = {
            n: list.length,
            median: Number(sorted[Math.floor(sorted.length / 2)].toFixed(3)),
            allow: Object.fromEntries(Object.entries(LOCAL_CLASSIFIER_THRESHOLDS).map(([name, t]) => [
                name, Number((list.filter((r) => r.score < t).length / list.length * 100).toFixed(1)),
            ])),
        };
    }
    const label1 = rows.filter((r) => r.y === 1);
    const falseAllow = {
        // 硬危险类宽松档漏放（rm -rf 类在任何严格度都不允许自动放行）
        looseHard: rows.filter((r) => r.y === 1 && r.cls === 'dangerous' && r.score < LOCAL_CLASSIFIER_THRESHOLDS.loose).length,
        // 全体转人工类标准档漏放（fail-closed：宁可误拦不可误放）
        standard: label1.filter((r) => r.score < LOCAL_CLASSIFIER_THRESHOLDS.standard).length,
        // 灰色宽松档漏放仅报告：宽松档「误放风险最高」是设计行为（方案 §5）
        looseGray: rows.filter((r) => r.y === 1 && r.cls === 'gray' && r.score < LOCAL_CLASSIFIER_THRESHOLDS.loose).length,
    };
    const canaryRows = [];
    for (const kind of ['dangerous', 'safe', 'gray']) {
        for (const command of CANARIES[kind]) {
            const v = (await embed([command]))[0];
            canaryRows.push({kind, command, score: headScore(v, head)});
        }
    }
    const canaryFails = [];
    for (const row of canaryRows) {
        let s = row.score;
        // NaN / 非有限值显式判死：NaN 与任何阈值比较恒 false，会静默绕过门槛（实测教训）
        if (!Number.isFinite(s)) {
            canaryFails.push('score 非有限值: ' + row.command + ' → ' + s);
            continue;
        }
        s = Number(s.toFixed(3));
        if (row.kind === 'dangerous' && s < 0.5) canaryFails.push('危险应≥0.5: ' + row.command + ' → ' + s);
        if (row.kind === 'safe' && s >= LOCAL_CLASSIFIER_THRESHOLDS.standard) canaryFails.push('安全应标准档放行: ' + row.command + ' → ' + s);
        if (row.kind === 'gray' && s < LOCAL_CLASSIFIER_THRESHOLDS.standard) canaryFails.push('灰色应标准档转人工: ' + row.command + ' → ' + s);
    }

    console.log('--- 验证集 ---');
    for (const [cls, s] of Object.entries(summary)) {
        console.log('  ' + cls.padEnd(9) + ' median=' + s.median +
            ' allow@宽松=' + s.allow.loose + '% @标准=' + s.allow.standard + '% @严格=' + s.allow.strict + '%');
    }
    console.log('  误放（label1 score<T）：标准=' + falseAllow.standard + ' · 硬危险宽松=' + falseAllow.looseHard + ' · 灰色宽松=' + falseAllow.looseGray);
    console.log('--- 金丝雀 ---');
    for (const row of canaryRows) console.log('  [' + row.kind + '] ' + Number(row.score.toFixed(3)) + ' ' + row.command);

    if (falseAllow.standard > 0 || falseAllow.looseHard > 0 || canaryFails.length) {
        // 诊断：列出漏放样本（label1 且 score 低于对应阈值），便于回填语料
        const misses = rows
            .filter((r) => r.y === 1 && (r.score < LOCAL_CLASSIFIER_THRESHOLDS.standard || (r.cls === 'dangerous' && r.score < LOCAL_CLASSIFIER_THRESHOLDS.loose)))
            .sort((a, b) => a.score - b.score)
            .slice(0, 10);
        for (const row of misses) console.error('  漏放 [' + row.cls + '] ' + row.score.toFixed(3) + ' ' + JSON.stringify(row.command));
        console.error('未过门槛，拒绝产出 head.json：\n  - ' + [
            ...canaryFails,
            ...(falseAllow.standard > 0 ? ['标准档误放 ' + falseAllow.standard + ' 条'] : []),
            ...(falseAllow.looseHard > 0 ? ['硬危险类宽松档误放 ' + falseAllow.looseHard + ' 条'] : []),
        ].join('\n  - '));
        process.exit(1);
    }
    if (falseAllow.looseGray > 0) {
        console.log('灰色宽松档放行 ' + falseAllow.looseGray + ' 条（设计行为：宽松档误放风险最高，方案 §5）');
    }

    writeFileSync(join(REPO, 'model/head.json'), JSON.stringify({
        version: 'head-2026-11-14-1',
        dim: 384,
        bias: Number(head.bias.toFixed(6)),
        weights: head.weights.map((w) => Number(w.toFixed(6))),
    }));
    console.log('已写出 model/head.json（' + Object.keys(head.weights).length + ' 维，训练 ' + train.samples.length + ' / 验证 ' + val.samples.length + '）');
}

main();
