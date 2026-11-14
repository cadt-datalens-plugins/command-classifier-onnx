// 插件入口冒烟检查（npm test）：
// 1) main.js / dist/worker.js / src/*.js 语法自检（node:module 逐文件解析）；
// 2) 用模拟 cadt API 执行 main.js，确认 registerCommandClassifier 被同步调用。
// 背景：0.2.0 的 main.js 曾把 await 写进非 async 的 Promise executor，
// 应用加载插件时直接 SyntaxError，分类器从未注册，状态行误报「语义模型未安装」。
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
let failed = 0;

function fail(message) {
    failed++;
    console.error('FAIL', message);
}

// 1) 语法检查：
//    main.js / dist/worker.js 是应用经 new Function('cadt', code) 执行的脚本，
//    用同样的 new Function 路径解析（等价于应用侧执行前解析）；
//    src/*.js 是 ESM 源（打包 worker 用），用动态 import 检查。
for (const rel of ['main.js', 'dist/worker.js']) {
    const code = readFileSync(path.join(root, rel), 'utf8');
    try {
        new Function(code);
    } catch (e) {
        fail(`${rel} 语法错误：${e.message}`);
    }
}
for (const rel of ['src/engine.js', 'src/tokenizer.js']) {
    const url = 'file:///' + path.join(root, rel).replace(/\\/g, '/');
    try {
        await import(url);
    } catch (e) {
        fail(`${rel} 加载失败：${e.message}`);
    }
}

// 2) 模拟 cadt 环境执行入口
const registered = [];
const cadt = {
    registerCommandClassifier: (spec) => registered.push(spec),
    addCommand: () => {},
    flash: (m) => console.log('[flash]', m),
    readAsset: () => Promise.reject(new Error('冒烟检查不读取真实资产')),
    openVirtualDoc: () => {},
};
try {
    new Function('cadt', readFileSync(path.join(root, 'main.js'), 'utf8'))(cadt);
} catch (e) {
    fail(`main.js 执行抛错：${e.message}`);
}
if (registered.length !== 1) {
    fail(`应恰好注册 1 个命令分类器，实际 ${registered.length}`);
} else if (typeof registered[0].classify !== 'function') {
    fail('注册的分类器缺少 classify 函数');
} else {
    console.log('OK 分类器已注册：', registered[0].id, '/', registered[0].label);
}

if (failed) {
    console.error(`\n${failed} 项检查失败`);
    process.exit(1);
}
console.log('\n入口冒烟检查通过');
