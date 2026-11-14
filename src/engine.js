// src/engine.js：MiniLM 句向量池化 + 逻辑回归头打分（Node 训练与浏览器推理同一实现）。
// sentence-transformers 约定：对 last_hidden_state 做 attention-mask 加权平均池化 → 384 维句向量。

// mean pooling：非 padding token 的隐藏态均值
export function meanPool(lastHiddenState, attentionMask, hiddenSize) {
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
export function headScore(embedding, head) {
    let z = head.bias || 0;
    const weights = head.weights;
    for (let d = 0; d < weights.length; d += 1) z += weights[d] * embedding[d];
    return 1 / (1 + Math.exp(-z));
}

// 单样本 SGD 步（训练用，Node 侧）：返回新 bias
export function headSgdStep(embedding, head, y, lr, l2) {
    const p = headScore(embedding, head);
    const g = p - y;
    for (let d = 0; d < head.weights.length; d += 1) {
        head.weights[d] -= lr * (g * embedding[d] + l2 * head.weights[d]);
    }
    return p;
}
