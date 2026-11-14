// command-classifier-onnx：端侧命令语义分类器插件（方案 B 二期，docs/unfinished-plans/local-command-classifier.md §8）。
// MiniLM int8 ONNX（22MB，模型在插件目录由下载者承担）+ WordPiece 分词 + 逻辑回归头，
// 经 registerCommandClassifier 注册进自动审批链；推理在 blob Worker 里跑 onnxruntime-web（不卡 UI）。
// 任何加载失败 → 回退一期词表打分（不阻塞审批链；连续失败熔断由核心负责）。
(function () {
    'use strict';

    if (typeof cadt.registerCommandClassifier !== 'function') {
        try { cadt.flash('[命令语义分类器] 需要应用 ≥ 1.2.0（registerCommandClassifier 未就绪）'); } catch (e) {}
        return;
    }

    // ---------- 状态 ----------
    const state = {
        phase: 'idle',          // idle → loading → ready / failed
        worker: null,
        seq: 0,
        pending: new Map(),     // id → {resolve, reject, timer}
        error: '',
        lastMs: 0,
        modelInfo: '',
        initPromise: null,
    };

    // readAsset → 二进制（data URL → fetch → ArrayBuffer）
    async function readBinary(path) {
        const dataUrl = await cadt.readAsset(path);
        const blob = await (await fetch(dataUrl)).blob();
        return await blob.arrayBuffer();
    }

    async function readJson(path) {
        const dataUrl = await cadt.readAsset(path);
        return JSON.parse(await (await fetch(dataUrl)).text());
    }

    // ---------- Worker 化 ONNX 推理 ----------
    function ensureWorker() {
        if (state.initPromise) return state.initPromise;
        state.initPromise = (async () => {
            state.phase = 'loading';
            // 一期回退词表先行注册完成，这里做 ONNX 慢加载
            const [modelBuffer, wasmBuffer, ortBlobUrl, workerBlobUrl, tokenizerData, head] = await Promise.all([
                readBinary('model/model_quantized.onnx'),
                readBinary('ort/ort-wasm-simd.wasm'),
                cadt.readAsset('ort/ort.min.js'),
                cadt.readAsset('dist/worker.js'),
                readJson('model/tokenizer.json'),
                readJson('model/head.json'),
            ]);
            state.modelInfo = 'MiniLM int8 · head ' + (head.version || '?') + ' · ' + Math.round(modelBuffer.byteLength / 1048576) + 'MB';
            const worker = new Worker(URL.createObjectURL(new Blob([await (await fetch(workerBlobUrl)).text()], {type: 'text/javascript'})));
            state.worker = worker;
            await new Promise((resolve, reject) => {
                const timer = setTimeout(() => reject(new Error('Worker 初始化超时（60s）')), 60000);
                worker.onmessage = (e) => {
                    const msg = e.data || {};
                    if (msg.type === 'ready') { clearTimeout(timer); resolve(); return; }
                    if (msg.type === 'error') { clearTimeout(timer); reject(new Error(msg.message)); }
                };
                worker.onerror = (e) => { clearTimeout(timer); reject(new Error('Worker 异常：' + ((e && e.message) || 'unknown'))); };
                worker.postMessage({
                    type: 'init',
                    ortUrl: URL.createObjectURL(new Blob([await (await fetch(ortBlobUrl)).text()], {type: 'text/javascript'})),
                    modelBuffer,
                    wasmBuffer,
                    tokenizerData,
                    head,
                }, [modelBuffer, wasmBuffer]);
            });
            state.phase = 'ready';
            pumpMessages();
        })().catch((e) => {
            state.phase = 'failed';
            state.error = String((e && e.message) || e);
            state.worker = null;
            state.initPromise = null; // 允许下次 classify 重试
            throw e;
        });
        return state.initPromise;
    }

    function scoreInWorker(command) {
        return new Promise((resolve, reject) => {
            const id = ++state.seq;
            const timer = setTimeout(() => {
                state.pending.delete(id);
                reject(new Error('推理超时（8s）'));
            }, 8000);
            state.pending.set(id, {resolve, reject, timer});
            state.worker.postMessage({type: 'score', id, command});
        });
    }

    // Worker 消息泵（初始化完成后接管 onmessage）
    function pumpMessages() {
        const worker = state.worker;
        if (!worker) return;
        worker.onmessage = (e) => {
            const msg = e.data || {};
            if (msg.type === 'result') {
                const pending = state.pending.get(msg.id);
                if (pending) {
                    state.pending.delete(msg.id);
                    clearTimeout(pending.timer);
                    state.lastMs = msg.ms || 0;
                    pending.resolve({score: msg.score, tokens: msg.tokens});
                }
                return;
            }
            if (msg.type === 'error') {
                if (msg.id !== undefined) {
                    const pending = state.pending.get(msg.id);
                    if (pending) {
                        state.pending.delete(msg.id);
                        clearTimeout(pending.timer);
                        pending.reject(new Error(msg.message));
                    }
                }
            }
        };
        worker.onerror = (e) => {
            // Worker 崩溃：让所有挂起请求失败，回退词表打分
            for (const pending of state.pending.values()) {
                clearTimeout(pending.timer);
                pending.reject(new Error('Worker 崩溃：' + ((e && e.message) || 'unknown')));
            }
            state.pending.clear();
            state.phase = 'failed';
            state.error = 'Worker 崩溃';
            state.worker = null;
            state.initPromise = null;
        };
    }

    // ---------- 一期回退：词表打分 ----------
    const DANGER_PATTERNS = [
        {re: /\brm\s+(-[a-z]*r[a-z]*f|-rf|--recursive\s+--force)/i, w: 0.9, name: '递归强删'},
        {re: /\b(del|erase)\b[^|]*\/(f|s)\b/i, w: 0.85, name: '强删文件'},
        {re: /\b(rmdir|rd)\s+\/s/i, w: 0.85, name: '递归删目录'},
        {re: /remove-item[^|]*-recurse/i, w: 0.85, name: '递归删目录'},
        {re: /\bdd\b[^|]*of=\/dev\//i, w: 0.95, name: '裸设备写入'},
        {re: /\bmkfs\S*\b|\bformat\s+[a-z]:/i, w: 0.95, name: '格式化'},
        {re: /\breg\s+(add|delete)\b/i, w: 0.7, name: '注册表修改'},
        {re: /\|\s*(sh|bash|zsh|pwsh|powershell)\b/i, w: 0.75, name: '管道落地 shell'},
        {re: /\biex\b|invoke-expression|powershell\s+(-enc|-encodedcommand)/i, w: 0.8, name: '代码执行'},
        {re: /\b(taskkill|kill|pkill)\b[^|]*\/?(f|-9)/i, w: 0.6, name: '强杀进程'},
        {re: /\bshutdown\b|\brestart-computer\b/i, w: 0.8, name: '关机重启'},
        {re: /git\s+push\s+(-f|--force)/i, w: 0.55, name: '强推'},
        {re: /git\s+(reset\s+--hard|clean\s+-[a-z]*f)/i, w: 0.6, name: '丢弃工作区'},
        {re: /\bcurl\b[^|]*(-f|--form|-F)\b|\bcurl\b[^|]*-F\s+f=@/i, w: 0.7, name: '数据外传'},
    ];

    function fallbackScore(command) {
        const text = String(command || '');
        let score = 0;
        const names = [];
        for (const pattern of DANGER_PATTERNS) {
            if (pattern.re.test(text)) {
                score += pattern.w;
                names.push(pattern.name);
            }
        }
        return {score: Math.min(1, score), names};
    }

    // ---------- 注册分类器 ----------
    cadt.registerCommandClassifier({
        id: 'onnx-semantics',
        label: '语义分类器（ONNX）',
        // ctx: {cwd, toolName, strictness, threshold, sessionId}
        async classify(command, ctx) {
            try {
                await ensureWorker();
                const result = await scoreInWorker(command);
                return {score: Math.min(1, Math.max(0, result.score)), reason: 'MiniLM 语义嵌入判定（' + result.tokens + ' tok · ' + state.lastMs + 'ms）'};
            } catch (e) {
                // 加载/推理失败：回退词表打分，不抛（连续失败熔断由核心处理）
                state.error = String((e && e.message) || e);
            }
            const fb = fallbackScore(command);
            return {
                score: fb.score,
                reason: '词表回退命中：' + (fb.names.join('、') || '无') + (state.error ? '（ONNX：' + state.error.slice(0, 60) + '）' : ''),
            };
        },
    });

    // ---------- 状态命令 ----------
    cadt.addCommand({
        id: 'classifier-status',
        label: '端侧分类器（ONNX）状态',
        run() {
            const lines = [
                '插件：command-classifier-onnx 0.2.0（二期 · ONNX 推理）',
                '模型：' + (state.modelInfo || '尚未加载（首次分类时懒加载）'),
                '阶段：' + state.phase + (state.error ? '（' + state.error.slice(0, 120) + '）' : '') + (state.lastMs ? ' · 最近推理 ' + state.lastMs + 'ms' : ''),
                '',
                '语义判定路径：MiniLM int8（22MB，插件目录）→ WordPiece 分词 → 平均池化 → 逻辑回归头。',
                '推理在 blob Worker 内（onnxruntime-web wasm）；加载失败自动回退词表打分。',
                '',
                '验证方法：设置 → AI 助手 → 权限 → 高级配置，端侧分类器启用后在',
                '「测试命令」输入 rm -rf /tmp/x（应转人工）或 git status（应放行）。',
            ];
            cadt.openVirtualDoc('端侧分类器（ONNX）状态.txt', lines.join('\n'));
        },
    });

    // 模型元数据读取（README 演示保留）
    try {
        Promise.resolve(cadt.readAsset('model/manifest.json')).catch(() => {});
    } catch (e) {}
})();
