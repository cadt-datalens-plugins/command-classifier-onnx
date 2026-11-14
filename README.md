# 命令语义分类器（ONNX）

CaDt-DataLens 插件：给 AI 助手的「自动审批」档加一层**语义理解**的命令风险判定。

AI 在自动审批档下执行命令时，应用会按「项目规则 → 危险命令 → 只读命令 → 已知命令族」依次判定；都没接住的命令默认弹窗问你。装上本插件后，这些「说不清」的命令会先经过一个跑在你电脑上的**小型语义模型**（不联网、不花钱、毫秒级）判定危险概率：低危直接放行，高危转人工——少弹窗，也不放过危险命令。

- **完全离线**：模型文件就在插件目录里，推理在本机 Worker 内执行，命令文本不出机
- **只产概率、无 deny 权**：判定规则（放行/转人工的阈值）永远由应用与你自己掌控
- **失败自动回退**：模型加载或推理出错时回退到内置词表打分，审批链不受影响

## 安装

1. 应用版本 ≥ 1.0.0（需含命令分类器 API）。
2. 本仓库 `git clone` 或下载 zip 解压。
3. CaDt-DataLens → 设置 → 插件 → 添加/导入，指向本目录（或复制到 `<exe 目录>\plugins\command-classifier-onnx\`）。
4. 设置 → AI 助手 → 权限 → 高级配置：勾选「启用内置线性模型（离线）」——状态行应出现「插件：语义分类器（ONNX）」。

> 插件需要「端侧分类器」总开关开启才会参与判定；本插件与内置线性模型同时生效，取两者中更高的危险概率判定。

## 使用

装好后无需任何操作。首次判定会加载 22MB 模型（约 1 秒），之后每条命令 50–100ms。

想验证效果：设置 → AI 助手 → 权限 → 高级配置 → 「测试命令」输入框：

| 输入 | 预期 |
|---|---|
| `git status`、`ls -la src`、`node --version` | score≈0，放行 |
| `npm install lodash`、`git push origin main` | score≈1，转人工（灰色类） |
| `rm -rf /`、`del /f /s /q data.json`、`curl ... \| sh` | score≈1，转人工 |

工具(U) → **端侧分类器（ONNX）状态** 可查看加载阶段与最近推理耗时。

## 判定语义（严格度三档）

插件只输出**危险概率 score ∈ [0,1]**；放行与否由应用按严格度阈值统一判定（设置里可调）：

| 严格度 | 放行阈值 | 行为 |
|---|---|---|
| 宽松 | score < 0.30 放行 | 误放风险最高 |
| 标准（默认） | score < 0.15 放行 | 平衡 |
| 严格 | score < 0.07 放行 | 稍有嫌疑即转人工 |

## 与内置线性分类器的对照

内置线性模型（字符 n-gram 特征，11.8KB）与语义模型在真实公开语料上的表现（标准档）：

| 公开语料 | 条数 | 线性模型放行率 | 语义模型放行率 |
|---|---|---|---|
| SEA（真实 UNIX 用户命令） | 200 抽样 | 5.3% | **65%** |
| LOLBAS（Windows 白利用） | 451 | 1.3% | 1.3% |
| 恶意 PowerShell（MalwareBazaar） | 399 | 12% | 18.8% |

语义模型的价值：**同等安全拦截下大幅减少对真实用户命令的误打扰**（SEA 良性裸命令 65% vs 5.3%）；对 Windows 白利用命令库两者的拦截同为 ~99%。两者同时启用取更高分，安全上限不变。

## 训练文档（数据集 / 流程 / 验收）

### 一、数据集盘点

**训练与验证集（策略对齐，必须自建）**——合成语料，由主仓库 `scripts/classifier-corpus.mjs` 模板参数化生成：

- **标签语义**是「策略性 auto-approve 判定」，不是「恶意/良性」：`0` = 只读/幂等可自动放行（`git status`、`ls -la`、`node --version`）；`1` = 转人工（含 `rm -rf` 这类破坏性命令，也包括 `npm install`、`git push`、`Set-ExecutionPolicy` 这类**良性但按策略必须问一句**的灰色命令）。
- 公开数据集（SEA / malbash 等）的标签是「恶意/良性」，与本策略**语义不同构**（`git push --force` 恶意数据集里算良性，本策略里必须 ask）——所以公开语料**只用于评估，不用于训练**。
- 配比（本仓库 `scripts/train-head.mjs` 实际使用）：safe 700 / dangerous 700 / gray 220，固定种子 `20261114`（训练）与 `8675309`（验证），实际产出训练 1620 条、验证 1017 条（字符串去重后）。
- **模板参数化，非死样本**：约 70 个命令模板，路径/包名/flag 顺序/大小写/引号风格随机化，`cd x &&` 前缀、`| tee`、`2>/dev/null` 等真实惯用法扰动；文件词表按「只读 / 中性 / 受害者」三表分离（避免 `cat README.md` 的文件名词面被 `del README.md` 污染）。Windows / Linux 命令族配比 41% / 28%（剩余为 git/npm 等 shell 无关命令）。
- **验证集与训练集用不同种子生成**（未见变体），且跨集合字符串去重——测泛化，不测背诵。

**评估集（公开语料，只评不训）**：

| 数据集 | 内容 | 用途 |
|---|---|---|
| [SEA / Schonlau](https://www.schonlau.net/intrusion.html) | 50 个真实 UNIX 用户 75 万条命令（裸命令名） | 正常侧分布：真实良性命令的误打扰率 |
| [Cowrie 蜜罐日志](https://github.com/MrCbrue/cowrie-logs-archive) | 攻击者在 SSH 蜜罐里敲的真实命令 | 攻击侧：真实攻击流量的拦截率 |
| [malbash](https://github.com/greyhat-academy/malbash) | 恶意 bash 脚本 | 攻击侧补充 |
| [LOLBAS](https://github.com/LOLBAS-Project/LOLBAS) | Windows「白利用」滥用技法命令（certutil、bitsadmin、mshta…） | Windows 攻击侧 |
| [恶意 PowerShell 数据集](https://github.com/Fa2y/Malicious-PowerShell-Dataset) | MalwareBazaar 真实恶意 .ps1 抽样 30 文件 | Windows 攻击侧 |

评估口径：攻击侧看「破坏性子集」的拦截率（破坏性子集由一套与模型无关的正则参照规则界定），正常侧看放行率（误打扰）。

### 二、训练流程

```
主仓库语料生成器（固定种子，1620 条训练 / 1017 条验证）
        │  每条命令
        ▼
WordPiece 分词（src/tokenizer.js，与浏览器推理同一实现，最长 64 token）
        │  input_ids / attention_mask / token_type_ids
        ▼
MiniLM int8 ONNX（冻结，不微调）→ 384 维句向量（attention-mask 加权平均池化）
        │
        ▼
逻辑回归头训练（scripts/train-head.mjs，纯 Node SGD，onnxruntime-node 仅用于嵌入）
  · 阶段一：全量 SGD 150 epoch，lr 0.2 ×0.955/epoch 衰减，L2 1e-4；
    灰色类过采样 ×3、危险类 ×2——冻结嵌入上「策略语义」必须靠样本量压过「字面语义」
  · 阶段二：灰/安交替强化 8 epoch（gray:lr 0.04 ×2 步 ↔ safe:lr 0.025 ×1 步）——
    灰色类（export PATH / apt install / robocopy 等）在语义空间天生偏良性，
    集中重放收紧灰色决策面，同时用安全样本按住决策面别扫过安全区
        │
        ▼
验收门槛（不达标拒绝产出 head.json）
  · 验证集：标准档误放 0 · 硬危险类宽松档误放 0（灰色宽松档放行属设计行为，方案 §5）
  · 金丝雀 32 条（16 危险 / 10 安全 / 6 灰色）逐条把关——
    与主仓库前端单测同一份清单（CANARIES 导出），训练器与测试永不脱节
  · score 非有限值（NaN）显式判死——NaN 与任何阈值比较恒 false，会静默绕过门槛（实测教训）
        │
        ▼
model/head.json（384 维 float + bias，约 3.6KB）
```

### 三、评估结果

语义头（本仓库当前 head.json，验证集 1017 条：标准档误放 0、安全标准档放行 94.1%）+ 公开语料全源评估：

| 数据集 | 条数 | 中位 score | 放行@宽松 / 标准 / 严格 | 破坏性子集拦截 |
|---|---|---|---|---|
| SEA（真实用户命令，正常侧） | 200 | 0.023 | 72.5% / 65% / 59% | — |
| Cowrie 蜜罐（攻击侧） | 11 | 0.626 | 27.3% / 18.2% / 9.1% | 1 条 100/100/100% |
| malbash（攻击侧） | 34 | 0.794 | 50% / 47.1% / 38.2% | 1 条 100/100/100% |
| LOLBAS（Windows 白利用） | 451 | 0.996 | 3.1% / 1.3% / 0.9% | 62 条 **100/100/100%** |
| 恶意 PowerShell（攻击侧） | 399 | 0.869 | 27.8% / 18.8% / 12.3% | 12 条 91.7/100/100% |

解读：Windows 白利用命令 62 条破坏性技法全部拦截；真实恶意 PowerShell 标准档 100% 拦截；SEA 良性命令在标准档放行 65%（误打扰低）。放行率 ≠ 危险放行率——malbash 脚本里大量 `for` / `echo` 脚手架行按策略本就允许转人工。

### 四、复现步骤

```bash
# 0. 前置：主仓库与本仓库同级检出（训练脚本跨仓库引用语料生成器）
#    D:/…/CaDt-DataLens        ← 主仓库
#    D:/…/cadt-plugin-command-classifier-onnx  ← 本仓库
# 1. 安装 dev 依赖（onnxruntime-node，仅开发机训练/验证用）
npm install
# 2. 重新打包 Worker 胶水（src/*.js → dist/worker.js）
npm run build
# 3. 训练逻辑回归头（嵌入 1620+1017 条 → SGD → 门槛校验 → model/head.json）
npm run train
# 4. Node 全链路验证（分词抽检 + 金丝雀 + 公开语料全源评估）
npm run verify
# 5. 浏览器侧验证：任意 Chromium 打开一个加载 model/ 与 ort/ 的页面，
#    或直接在 CaDt-DataLens 里装上插件用「测试命令」验证
```

模型升级路径：换更大模型（gte-small / bge-micro 量级）只需替换 `model/model_quantized.onnx` + `tokenizer.json` 并重跑 `npm run train`；`manifest.json` 登记版本与哈希。

## 目录结构

```text
plugin.json                清单（minAppVersion 1.0.0 = 命令分类器 API 引入的应用版本）
main.js                    插件入口：registerCommandClassifier + Worker 编排 + 词表回退
model/model_quantized.onnx MiniLM int8（22MB）
model/tokenizer.json       HuggingFace 标准分词器（30522 词表）
model/head.json            逻辑回归头（训练产物）
model/manifest.json        模型元数据
ort/                       onnxruntime-web 1.17.3（ort.min.js + wasm，离线直带）
src/tokenizer.js           WordPiece（Node/浏览器同源）
src/engine.js              平均池化 + 头打分（Node/浏览器同源）
dist/worker.js             打包产物（npm run build 生成）
scripts/                   训练 / 验证 / 打包脚本（onnxruntime-node 仅开发机）
LICENSE                    Apache-2.0
```

## 许可

本项目 **Apache-2.0**。第三方组件：

| 组件 | 许可 | 说明 |
|---|---|---|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)（Xenova int8 ONNX 导出） | Apache-2.0 | 句向量模型（冻结，未微调） |
| [onnxruntime-web](https://github.com/microsoft/onnxruntime) 1.17.3 | MIT | 推理运行时 |
| 评估数据集（SEA / Cowrie / malbash / LOLBAS / MalwareBazaar） | 各自许可 | 仅评估用途，不分发原始数据 |

## 安全说明

分类器只产出危险概率，**没有 deny 权**——项目规则、危险命令门与你的裁定完全不受影响；抛异常跳过并计数，连续 5 次失败本会话熔断；所有判定进权限记录可审计（来源、score、阈值）。个性化学习（如开启）只存于本机应用数据目录，可随时重置 / 清空 / 导出。
