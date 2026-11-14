# 命令语义分类器（ONNX）插件

端侧命令风险分类器（方案 B，主仓库 [docs/unfinished-plans/local-command-classifier.md](https://gitee.com/cadt-datalens-plugins) §8）：向自动审批链注册一个**语义级**分类器，与内置线性分类器（方案 A）**保守合并取最高分**——任一分类器判险即转人工，`deny` 权永远不下放给模型。

## 二期：真实 ONNX 推理（本仓库当前状态）

- **模型**：sentence-transformers/all-MiniLM-L6-v2（Xenova 导出 int8 量化 ONNX，**22 MB**，本仓库 `model/` 内直带，下载者承担体积）
- **链路**：WordPiece 分词（`src/tokenizer.js`，Node/浏览器同一实现）→ MiniLM 嵌入 → 平均池化 → 逻辑回归头（`model/head.json`，主仓库语料训练）→ 危险概率
- **运行时**：onnxruntime-web 1.17.3（`ort/`，wasm SIMD），推理在 **blob Worker** 内执行（不卡 UI）；`wasmBinary` 直传，无 CDN 依赖、完全离线
- **回退**：模型加载或推理失败自动回退一期词表打分，不阻塞审批链；连续失败熔断由应用核心负责

## 语义头训练指标（验证集 576 条，金丝雀门槛）

训练管线：主仓库语料（训练 930 / 验证 576，种子固定可复现）→ 嵌入 → 逻辑回归 SGD（40 epoch）。
门槛：标准档误放 0、危险金丝雀 ≥0.5 且宽松档必拦、安全金丝雀标准档放行、灰色金丝雀标准档转人工——与线性模型同一验收纪律。指标以最近一次 `npm run train` 输出为准（写入 `model/head.json` 前强校验）。

## 本地测试

1. 应用 ≥ 1.2.0（需含 `registerCommandClassifier` API）。
2. 设置 → 插件 → 添加/导入 本目录（或复制到 `<exe 目录>\plugins\command-classifier-onnx\`）。
3. 设置 → AI 助手 → 权限 → 高级配置：启用「端侧分类器」，状态行应出现「插件：语义分类器（ONNX）」。
4. 「测试命令」输入 `rm -rf /tmp/x`（应转人工）/ `git status`（应放行）；首次判定会懒加载 22MB 模型，之后毫秒级。
5. 工具(U) → 端侧分类器（ONNX）状态：查看加载阶段 / 最近推理耗时。

## 目录结构

```text
plugin.json              清单（minAppVersion 锁 1.2.0）
main.js                  入口：registerCommandClassifier + blob Worker 编排 + 词表回退
model/model_quantized.onnx   MiniLM int8（22MB）
model/tokenizer.json     HF 标准分词器
model/head.json          逻辑回归头（scripts/train-head.mjs 产出）
model/manifest.json      模型元数据
ort/ort.min.js + *.wasm  onnxruntime-web 1.17.3（离线直带）
src/tokenizer.js         WordPiece（Node/浏览器同源）
src/engine.js            池化 + 头打分（Node/浏览器同源）
dist/worker.js           打包产物（build 脚本生成）
scripts/                 训练 / 验证 / 打包（dev 依赖 onnxruntime-node，仅开发机用）
```

## 开发

```bash
npm install        # dev 依赖（onnxruntime-node）
npm run build      # 重新打包 dist/worker.js
npm run train      # 训练逻辑回归头（需主仓库同级检出：../CaDt-DataLens）
npm run verify     # Node 全链路验证（分词抽检 + 金丝雀 + 公开语料抽样 eval）
```

## 上架

应用「设置 → 插件 → 插件市场」列出 Gitee 组织 `cadt-datalens-plugins` 下公开仓库；本仓库推送即上架（仓库根 `plugin.json` 提供元数据）。更新模型 / 头时同步 bump `version` 与 `manifest.json` 哈希登记。

## 安全说明

分类器只产出危险概率，没有 deny 权；项目规则、危险命令门与用户裁定完全不受影响。抛异常跳过并计数熔断。模型与头随仓库分发（哈希见 manifest），推理全部在本机 Worker 内执行，命令文本不出机。
