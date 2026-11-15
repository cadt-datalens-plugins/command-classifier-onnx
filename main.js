// command-classifier-onnx：端侧命令语义分类器插件（方案 B 二期，docs/unfinished-plans/local-command-classifier.md §8）。
// MiniLM int8 ONNX（22MB，模型在插件目录由下载者承担）+ WordPiece 分词 + 逻辑回归头，
// 经 registerCommandClassifier 注册进自动审批链；推理在 blob Worker 里跑 onnxruntime-web（不卡 UI）。
// 任何加载失败 → 回退一期词表打分（不阻塞审批链；连续失败熔断由核心负责）。
(function () {
    'use strict';

    if (typeof cadt.registerCommandClassifier !== 'function') {
        try { cadt.flash('[命令语义分类器] 应用版本过低：请更新到含命令分类器 API 的新版本'); } catch (e) {}
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
            // ort.min.js 同样转 blob URL；必须在 executor 外构造（executor 是普通函数，
            // 里面写 await 是语法错误——曾导致整个插件加载失败、状态行误报「未安装」）
            const ortUrl = URL.createObjectURL(new Blob([await (await fetch(ortBlobUrl)).text()], {type: 'text/javascript'}));
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
                    ortUrl,
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
                    // worker 结果里的 segs（段数）与 truncated（窗口溢出）必须透传：
                    // classify 读 result.segs 拼 reason、读 result.truncated 走 fail-closed。
                    pending.resolve({score: msg.score, tokens: msg.tokens, segs: msg.segs, truncated: msg.truncated});
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

    // 一期词表回退已移除（红队 P0-2：14 条黑名单式正则未命中返回 0 分、32 类危险命令全档放行；
    // ONNX 失败现改为抛错触发核心熔断，由线性模型与人工兜底）。

    // ---------- 注册分类器 ----------
    cadt.registerCommandClassifier({
        id: 'onnx-semantics',
        label: '语义分类器（ONNX）',
        // ctx: {cwd, toolName, strictness, threshold, sessionId}
        async classify(command, ctx) {
            try {
                await ensureWorker();
                const result = await scoreInWorker(command);
                if (result.truncated) {
                    // 红队 P0 兜底：单段超 62 token 的命令 fail-closed（worker 已置 score=1）
                    return {score: 1, reason: '命令超长（分段推理仍溢出窗口），fail-closed 转人工'};
                }
                return {score: Math.min(1, Math.max(0, result.score)), reason: 'MiniLM 语义嵌入判定（' + result.segs + ' 段 · ' + state.lastMs + 'ms）'};
            } catch (e) {
                // 红队 P0-2：加载/推理失败必须抛错触发核心熔断——静默回退词表会让 32 类
                // 危险命令（curl -d 外传 / chmod -R 777 / LOLBAS 等）得 0 分全档放行，
                // 且失败计数永不增长、系统长期停留在不安全降级态。
                state.error = String((e && e.message) || e);
                throw new Error('语义模型不可用：' + state.error);
            }
        },
    });

    // ---------- 状态命令 ----------
    cadt.addCommand({
        id: 'classifier-status',
        label: '端侧分类器（ONNX）状态',
        run() {
            const lines = [
                '插件：command-classifier-onnx 0.2.2（二期 · ONNX 推理）',
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
