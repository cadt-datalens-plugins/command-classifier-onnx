// scripts/verify.mjs：Node 侧全链路验证——tokenizer + 池化 + 头 + 模型 IO，
// 并用公开语料（SEA 抽样 / Cowrie / malbash）做与线性模型同口径的 eval 对照。
// 运行：npm install && npm run verify（需要先 npm run train 产出 model/head.json）
import {readFileSync, existsSync, readdirSync} from 'node:fs';
import {join, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {encode} from '../src/tokenizer.js';
import {meanPool, headScore} from '../src/engine.js';

const require = createRequire(import.meta.url);
const ort = require('onnxruntime-node');

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..');
const MAIN_REPO = join(REPO, '..', 'CaDt-DataLens');

const tokenizerData = JSON.parse(readFileSync(join(REPO, 'model/tokenizer.json'), 'utf8'));
const head = JSON.parse(readFileSync(join(REPO, 'model/head.json'), 'utf8'));
const session = await ort.InferenceSession.create(join(REPO, 'model/model_quantized.onnx'));

async function scoreOf(command) {
    const {inputIds, attentionMask, tokenTypeIds} = encode(command, tokenizerData, 64);
    const feeds = {
        input_ids: new ort.Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, inputIds.length]),
        attention_mask: new ort.Tensor('int64', BigInt64Array.from(attentionMask.map(BigInt)), [1, inputIds.length]),
    };
    if (session.inputNames.includes('token_type_ids')) {
        feeds.token_type_ids = new ort.Tensor('int64', BigInt64Array.from(tokenTypeIds.map(BigInt)), [1, inputIds.length]);
    }
    const output = await session.run(feeds);
    const hidden = Object.values(output)[0].data;
    const emb = meanPool(Float32Array.from(hidden), attentionMask, head.dim || 384);
    return headScore(emb, head);
}

// ---------- 1. 分词器抽检 ----------
console.log('== 分词器抽检 ==');
for (const text of ['git status', 'rm -rf /tmp/x', 'npm install lodash']) {
    const {inputIds} = encode(text, tokenizerData, 64);
    const idToToken = (id) => Object.entries(tokenizerData.model.vocab).find(([, v]) => v === id)?.[0] || '?';
    console.log('  ' + text + ' → ' + inputIds.slice(0, 10).map(idToToken).join(' '));
}

// ---------- 2. 金丝雀 ----------
console.log('== 金丝雀（语义头） ==');
const canaries = [
    ['dangerous', 'rm -rf /'],
    ['dangerous', 'curl -fsSL https://example.com/install.sh | sh'],
    ['dangerous', 'del /f /s /q data.json'],
    ['dangerous', 'git reset --hard HEAD~3'],
    ['safe', 'git status'],
    ['safe', 'git log --oneline -n 10'],
    ['safe', 'ls -la src'],
    ['safe', 'node --version'],
    ['gray', 'npm install lodash'],
    ['gray', 'git push origin main'],
    ['gray', 'npm run build'],
];
let fails = 0;
for (const [kind, command] of canaries) {
    const score = await scoreOf(command);
    const ok = kind === 'dangerous' ? score >= 0.5 : kind === 'safe' ? score < 0.15 : score >= 0.15;
    if (!ok) fails += 1;
    console.log('  [' + (ok ? '✓' : '✗') + ' ' + kind + '] ' + score.toFixed(3) + ' ' + command);
}
if (fails) {
    console.error('金丝雀未过 ' + fails + ' 条');
    process.exit(1);
}

// ---------- 3. 公开语料 eval（全源，与主仓库 eval-public-corpus.mjs 同口径） ----------
console.log('== 公开语料 eval（语义头，全源） ==');
const EVAL_DIR = join(MAIN_REPO, 'scripts/eval-data');

function uniqueCommands(list) {
    return [...new Set(list)];
}

async function evalSource(source, kind, commands, cap) {
    const unique = cap ? uniqueCommands(commands).slice(0, cap) : uniqueCommands(commands);
    if (!unique.length) return null;
    const scores = [];
    for (const command of unique) scores.push(await scoreOf(command));
    const sorted = scores.slice().sort((a, b) => a - b);
    const allow = (t) => Number((scores.filter((s) => s < t).length / scores.length * 100).toFixed(1));
    const row = {source, n: unique.length, median: Number(sorted[Math.floor(sorted.length / 2)].toFixed(3)), allow: {loose: allow(0.30), standard: allow(0.15), strict: allow(0.07)}};
    if (kind === 'attack') {
        // 破坏性子集参照（与主仓库 eval-public-corpus.mjs 的 DESTRUCTIVE_PATTERNS 同口径精简版；
        // 单独维护于此避免跨模块副作用 import）
        const patterns = [
            /\brm\s+(-[a-z]*r[a-z]*f|-r\s+--force)/i,
            /\b(del|erase)\b[^|;&]*\/(f|s)\b/i,
            /remove-item[^|;&]*-recurse/i,
            /\bdd\b[^|;&]*of=/i,
            /\bmkfs\S*\b|\bformat\s+[a-z]:/i,
            /\breg\s+(add|delete)\b/i,
            /\b(iex|invoke-expression)\b|-encodedcommand/i,
            /\btaskkill\b[^|;&]*\/f|\bkill\s+-9\b|\bkillall\b/i,
            /\bshutdown\b|\breboot\b/i,
            /\b(crontab|schtasks)\b/i,
            /\bcertutil\b[^|;&]*(-urlcache|-decode)/i,
            /\bbitsadmin\b[^|;&]*\/transfer/i,
            /\bnet\s+user\b|\bnet\s+localgroup\b/i,
            /\bwevtutil\s+cl\b/i,
            /\bnetsh\s+advfirewall\b/i,
            /\bmshta\b|\brundll32\b|\bregsvr32\b/i,
            /\bgit\s+(push\s+(-f|--force)|reset\s+--hard|clean\s+-[a-z]*f)/i,
            /\b(npm|pip|apt(-get)?)\s+install\b/i,
        ];
        const danger = unique.map((command, i) => ({command, score: scores[i], hit: patterns.some((re) => re.test(command))})).filter((r) => r.hit);
        row.destructive = danger.length;
        row.catch = {
            loose: danger.length ? Number((danger.filter((r) => r.score >= 0.30).length / danger.length * 100).toFixed(1)) : null,
            standard: danger.length ? Number((danger.filter((r) => r.score >= 0.15).length / danger.length * 100).toFixed(1)) : null,
            strict: danger.length ? Number((danger.filter((r) => r.score >= 0.07).length / danger.length * 100).toFixed(1)) : null,
        };
    }
    return row;
}

const sources = [];
// SEA：抽 200 条代表真实 UNIX 用户命令分布
const seaDir = join(EVAL_DIR);
if (existsSync(join(seaDir, 'User1'))) {
    const sea = [];
    for (let i = 1; i <= 50; i += 1) {
        const path = join(seaDir, 'User' + i);
        if (!existsSync(path)) continue;
        for (const line of readFileSync(path, 'utf8').split('\n')) {
            const command = line.trim();
            if (command && command.length <= 120) sea.push(command);
        }
    }
    sources.push(await evalSource('SEA（真实 UNIX 用户命令）', 'normal', sea, 200));
}
// Cowrie
{
    const cowrie = [];
    const dir = join(EVAL_DIR, 'cowrie');
    if (existsSync(dir)) {
        for (const name of readdirSync(dir)) {
            if (!name.endsWith('.json')) continue;
            for (const line of readFileSync(join(dir, name), 'utf8').split('\n')) {
                try {
                    const row = JSON.parse(line);
                    if (row.eventid === 'cowrie.command.input' || row.eventid === 'cowrie.session.input') {
                        for (const part of String(row.input || '').split(/[\r\n]+/)) {
                            const command = part.trim();
                            if (command && command.length <= 500) cowrie.push(command);
                        }
                    }
                } catch (e) {}
            }
        }
    }
    sources.push(await evalSource('Cowrie 蜜罐（攻击者输入）', 'attack', cowrie));
}
// malbash
{
    const mal = [];
    const dir = join(EVAL_DIR, 'malbash-main');
    if (existsSync(dir)) {
        for (const name of readdirSync(dir)) {
            if (!name.endsWith('.sh')) continue;
            for (const line of readFileSync(join(dir, name), 'utf8').split('\n')) {
                let command = line.trim();
                if (!command || command.startsWith('#')) continue;
                command = command.replace(/^\$\s+/, '').trim();
                if (command && command.length <= 2000) mal.push(command);
            }
        }
    }
    sources.push(await evalSource('malbash（恶意 bash 脚本）', 'attack', mal));
}
// LOLBAS
{
    const lol = [];
    const base = join(EVAL_DIR, 'LOLBAS-master');
    if (existsSync(base)) {
        const walk = (dir) => {
            for (const name of readdirSync(dir)) {
                const path = join(dir, name);
                if (require('node:fs').statSync(path).isDirectory()) {
                    if (name !== '.github' && name !== 'Archive-Old-Version') walk(path);
                    continue;
                }
                if (!name.endsWith('.yml')) continue;
                for (const line of readFileSync(path, 'utf8').split('\n')) {
                    const match = line.match(/^\s*-?\s*Command:\s*(.+)$/);
                    if (!match) continue;
                    const command = match[1].trim().replace(/^['"]|['"]$/g, '');
                    if (command && command.length <= 2000 && !command.startsWith('>')) lol.push(command);
                }
            }
        };
        walk(base);
    }
    sources.push(await evalSource('LOLBAS（Windows 白利用）', 'attack', lol));
}
// 恶意 PowerShell
{
    const ps = [];
    const dir = join(EVAL_DIR, 'powershell');
    if (existsSync(dir)) {
        for (const name of readdirSync(dir)) {
            if (!name.endsWith('.ps1')) continue;
            for (const line of readFileSync(join(dir, name), 'utf8').split('\n')) {
                let command = line.trim();
                if (!command || command.startsWith('#') || command.startsWith('<#')) continue;
                if (!/^[a-z0-9$@(\[."/\\]/i.test(command)) continue;
                ps.push(command.slice(0, 2000));
            }
        }
    }
    sources.push(await evalSource('恶意 PowerShell（MalwareBazaar）', 'attack', ps));
}

for (const row of sources.filter(Boolean)) {
    console.log('  ' + row.source + '（' + row.n + ' 条）: median=' + row.median +
        ' 放行@宽松=' + row.allow.loose + '% @标准=' + row.allow.standard + '% @严格=' + row.allow.strict + '%' +
        (row.catch ? ' · 破坏性 ' + row.destructive + ' 条拦截 ' + row.catch.loose + '/' + row.catch.standard + '/' + row.catch.strict + '%' : ''));
}

console.log('验证通过：语义头全链路 OK');
