# EgoConseq

## 代码与数据获取

当前核心代码位于 [EgoConseq-Bench/P_bench](https://github.com/syp2ysy/EgoConseq-Bench/tree/main/P_bench)。
完整 `data/` 快照单独存放在
[syp115/EgoConseq-Bench](https://huggingface.co/datasets/syp115/EgoConseq-Bench)，
**上传状态：** 数据尚未完成上传；所有分卷到齐后再执行下方恢复命令。

包括 Benchmark、SFT、源记录、已保存的 response 和评测产物；不包含模型权重或外部模拟器、源数据集安装。
归档保留内部软链接及硬链接，不改变题目、GT 或图像。

在本项目 `P_bench/` 目录执行：

```bash
hf auth login  # 访问私有数据仓库时需要登录
hf download syp115/EgoConseq-Bench --repo-type dataset \
  --include 'data.tar.zst.part-*' SHA256SUMS archive_manifest.json \
  --local-dir ../egoconseq-data-download
(cd ../egoconseq-data-download && sha256sum -c SHA256SUMS)
cat ../egoconseq-data-download/data.tar.zst.part-* | zstd -d | tar -xf -
```

请在全新 checkout 中恢复：解压会重建 `data/` 并覆盖同路径文件，所有分卷均不可缺少。
归档大小和清单见 Hugging Face 数据说明。将 `.env.example` 复制为 `.env.local`，
填写本机环境和源数据路径，再执行 `source .env.local`；各模型的环境要求见下方 inference README。
采集元数据可能保留原机器的绝对源路径，重新采集需要配置对应的外部源数据。

`environment/habitat.yml` 中的采集环境使用 Python 3.9；推理和 LLM 评测需要 Python 3.10
或更新版本，并安装对应模型依赖。采集与构建测试使用 Habitat 环境，judge 测试使用安装了
NumPy 和 pytest 的 Python 3.10+ 环境：

```bash
"${EGOCONSEQ_HABITAT_PYTHON:-python}" -m pytest --ignore=tests/test_pl_judge_evaluation.py
"${EVAL_PYTHON:-python}" -m pytest tests/test_pl_judge_evaluation.py
```

恢复数据后，使用 `python visualization/plot_benchmark_figures.py` 生成可编辑统计图；
已保存的 SVG/PDF/PNG 图及统计数值位于 `output/figures/paper_20260911/`。

从初始第一视角 RGB 图像预测机器人动作后果。当前 seen 数据为 **58,557 条
compact records**：B1K 11,750、GS 24,569、R2R 22,238。B1K 已按要求停止扩采，
保留所有已接受记录并更新 manifest、数量和场景索引。

## 正式目录

```text
data/metadata/train/seen_updates/current/   原 records 和原图
data/benchmark/
  benchmark/seen/{QA*.json,images/}          每个输入版本 4,999 QA，共用图片
  benchmark/unseen/{QA*.json,images/}        每个输入版本 2,000 QA，共用图片
  metadata/frozen.json                      冻结摘要及全部 7,000 条 SFT 排除 records
  metadata/{seen,unseen}/                   record_index.json 和 report.json
  inference/                               Qwen/Cosmos 推理代码、启动脚本
    responses/                             当前保存的 QA 对应的各模型 response
    logs/                                  运行日志
  index.html                               唯一主页，支持 All/Seen/Unseen
```

构建 benchmark 只读 records，不改 pose、半径、高度、FOV、actions 或 GT。
原先从 5,000 条不同 record 各选一道题；**撤销 1 道 A3 后，SFT 仍排除原先这 5,000 条 record 的所有 QA**。
剩余 record 仍可贡献多个任务、动作和表面点的训练题，不是一条 record 一道 SFT。

经授权撤销有疑点的 A3 后，当前 benchmark 为 Seen 4,999 + Unseen 2,000，共 6,999 道八任务 QA。
本轮只撤销 40 个 case 的 A3 输出及对应的 1 道 benchmark、16 道 SFT 问题；
其他 record 字段和保留的 QA 不变。目录名只表示 split，不再包含数量，SFT 隔离名单继续保留。
`metadata/frozen.json` 记录两份 QA、私有索引及图片路径/摘要集合的哈希，同时保存全部
7,000 条 record UID 供 SFT 整条排除。不要修改 GT 或重新选题；题面修订须经明确
授权并更新冻结哈希。
HTML 展示仍可更新，入口统一为 `http://localhost:8000/`。冻结不复制原图或 records。

Benchmark 和 SFT 统一使用 `post_QA/templates.py::SYSTEM_PROMPT`
（`abc1-prompts-v8-clear-questions`）：System 定义移动机器人角色、顺序执行的
前进与转向、圆形碰撞占地及相机朝向，明确
“Follow the listed motions exactly, without changing the path.”，地面假设为
“Assume continuous ground.”。机体半径、相机光心高度和水平／垂直视场角
统一放在 User 的 Configuration 段；八个任务各有十个简洁问题模板，与 SFT 共用。
`refresh-prompts --keep-templates` 同步题面并保持原有模板分配、图片、GT 和选样，
哈希变更记录在 `frozen.json` 的 `prompt_updates` 中。

相机高度使用**采集时设置的、相对机器人局部地面参考的垂直高度**，直接读取
`sensor.nominal_camera_offset_m`，只有 **0.5 / 1.0 / 1.5 m** 三档，不使用事后
计算的场景网格距离或拟合平面距离。`repair-parameters` 同步冻结索引、Benchmark、
SFT（包括已导出的参数消融版本）和 HTML 中的高度及说明文字；原始 records、
图片、pose、动作、半径、FOV 和 GT 均不改动。最新同步统计原地覆盖
`data/benchmark/metadata/parameter_audit.json`，不额外保存副本。

## 使用

### 对已保存的 response 评分

```bash
bash data/benchmark/evaluation/run_eval.sh \
  --responses data/benchmark/inference/responses/cosmos3-edge_thinking_off_response.json
```

除 A3 外，明确答案用正则提取，长回答由不接收 GT 的 LLM 提取，再按统一规则评分，
距离同时报告 **±0.25 m 和 ±0.5 m**。A3 的每条回答与 GT 交给 LLM 判断语义是否一致，
汇总整个 A3 的准确率。

`--rules-only` 可不加载 LLM；去掉该参数、运行同一命令即可继续提取剩余回答。
默认提取模型是本地缓存的 Qwen3-8B；已有 API 可使用
`--extractor-model MODEL --base-url URL`，密钥变量为 `INFER_API_KEY`。
逐题结果和独立分数汇总保存在 `data/benchmark/evaluation/results/<student>/`。
具体规则见 [evaluation/README.md](data/benchmark/evaluation/README.md)。

### 编译数据

```bash
HABITAT_PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python

$HABITAT_PY scripts/build_seen_benchmark.py compile \\
  --output outputs/benchmark-build/new_run --seed 20260906

$HABITAT_PY scripts/build_seen_benchmark.py compile-sft \\
  --output data/sft/seen_v2 \\
  --exclude-index data/benchmark/metadata/frozen.json

```

编译工作产物放在 `outputs/benchmark-build/`，与已冻结的正式包分开；不要覆盖当前发布版本。

只扫描一次结构化 records，先保留各长度的 case/point 引用，再按分层候选批次
检查图像质量、计算 DINOv2-S/14 特征。缺额才扩大候选批次，不重新编码已有图。
全部选中初始图跨任务、跨数据集比较 DINO cosine；特征仅存在内存，不写入 records。
C1 四个终点选项检查质量，不强求四个选项之间语义差异很大。

A4/B1/B2/B3 使用无编号红点加黑白双描边，点心就是原始固定表面点。以下命令
只从保存的 QA 和元数据重建 HTML，不改原图、题面、选样或 GT，不跑 DINO：

```bash
$HABITAT_PY scripts/build_seen_benchmark.py browser --root data/benchmark
```

任务总数见 [seen_5000_v1.json](post_QA/specs/seen_5000_v1.json)。普通任务和 C1
尽量均匀覆盖 L1–L6；A2 均衡 L3–L6，不生成 L3 Turn-first。
A2 在每个 length/start 内平衡 Forward ordinal 和距离 rank。
每个受支持 dataset/task 至少 30% Turn-first；场景、答案、动作形态轮换选取。
实际长度分配、答案统计和最相近图对保存在 report/HTML，不能把候选不足宣称为达标。

### Unseen benchmark

复用同一编译器，通过 `--spec post_QA/specs/unseen_2000_v1.json` 选择 2,000 题目标。
每题使用不同 record；A2 仅 L3–L6、L3 无 Turn-first，最终长度比例按图像合格供给调整。
GS 不生成 A3/B1。全部 unseen records 仅用于评测，不进入 SFT。

`plan --supply-report` 可以只补缺失的 A2 类别；补采读取原场景 split 和 manifest，
不改 pose、半径、高度或 FOV。`update-records --benchmark-output <work-output>`
会在合并后自动尝试编译 QA、私有 record 索引和 `index.html`；不指定这个参数时仍只更新 records。
补采工作目录为 `data/metadata/test/unseen_updates/`，最新 records 由其中 `current` 指向。
原始 `data/metadata/test/unseen/` 在新产物验收前保留。DINO 特征只存在内存中。

## 模型推理：只保存回答，不评分

每个 split 保留 `QA.json` 完整版，并提供 `QA_no_radius.json`、`QA_no_height.json`、
`QA_no_fov.json`、`QA_no_parameters.json`。与 SFT 一样，只删配置参数行：去 FOV 同时
删除 HFOV/VFOV，全部去掉则删除整段 Configuration；system、动作、问题、GT、ID、
顺序和图片引用均不变。使用 `run.sh <模型> --variant no_height` 等参数选择版本，
response 按版本独立命名。以后经授权更新完整版时，再执行
`scripts/build_seen_benchmark.py export-benchmark --root data/benchmark` 同步派生 JSON。

每个模型系列在 `data/benchmark/inference/` 中有独立的 `infer_*.py`；
`common.py` 只共用 QA 输入、HTTP 通信、保存与续跑。统一启动入口包含 14 个具身／空间模型
及 Qwen 对照配置，直接读取保存的 Seen/Unseen QA：

```bash
bash data/benchmark/inference/run.sh list
bash data/benchmark/inference/run.sh all --dry-run
bash data/benchmark/inference/run.sh qwen3vl-4b --limit-per-task 1
bash data/benchmark/inference/run.sh all
```

`all` 顺序运行，按现有 response 续跑并逐个释放显存；`--dry-run` 不下载、不加载模型。
默认读取两份 QA 共 6,999 题，`--limit-per-task 1` 则覆盖两个 split 八任务共 16 题，
包括 C1。权重共用 Hugging Face 缓存，不存入 benchmark。
Qwen 系复用 Transformers；SenseNova-SI 使用 vLLM；SpatioLM 调用官方自定义空间模型；
Cosmos3-Edge 使用 Reasoner，不使用 Policy-DROID。
详细命令与依赖见 [inference 使用说明](data/benchmark/inference/README.md)。
五个已缓存的小模型按顺序推理，再统一加载 Qwen3-8B 评分：

```bash
bash data/benchmark/inference/run.sh small --cached --serve --evaluate
```

`small` 包含 Qwen3-VL-4B、RoboBrain2.5-4B、RoboInter-3B、RynnBrain1.1-2B、
Cosmos3-Edge。使用当前 v8 题面并固定本地权重版本，不下载模型。
进度见 `data/benchmark/inference/pipeline_status.json`；续跑校验题目哈希、模型设置和评分协议。

输入组成保持不变：

- **System prompt：**直接使用 QA 保存的 `system`，定义机器人角色、连续执行、原地转向、
  圆形碰撞占地和相机朝向，以及所有任务共享的连续
  地面假设；推理时不追加隐藏指令。
- **User：**完整原题面，包括底盘半径、相机高度、HFOV/VFOV、编号动作、
  具体问题，以及该题已有的方向约定。
- **图像：**路径相对于所在 split 的 `QA.json` 解析，逐个原位替换
  `<image>`。A1–B3 为一张图，A4/B1/B2/B3 保留现有标记点图。
  C1 在同一条 user 消息中交错输入：**初始图→问题与 A 标签→A 图→B 标签→
  B 图→C 标签→C 图→D 标签→D 图**。不拼图、不重排、不重新编号。
  仅使用模型自带图像预处理；原生适配器设置记录在输出，API 服务启动参数保存在对应日志。
- **GT：**原有 `assistant` 消息完全不传给模型。

每个任务保留十个简洁、语义等价的模板，回答格式只写在对应题面。
A2/A3 明确会发生碰撞，A2 使用完整动作列表编号；A4/B1/B2/B3/C1 明确序列无碰撞。
只有 C1 要求回答 A/B/C/D。这次 prompt 更新不改变图片、几何参数、题目 ID 或 GT。
A1 直接问执行中是否碰撞，不重复 endpoint 或底盘定义。A4/B3 用独立的
`Query moment` 行保留动作序号与行进距离百分比；距离题仍问相机光心到点的
三维直线距离，方向题仍采用原来的 24 类边界。

需要经授权同步题面时运行：

```bash
$HABITAT_PY scripts/build_seen_benchmark.py refresh-prompts --keep-templates
```

该命令只读取已有索引和结构化输入，原地同步 Benchmark、SFT（含已导出的消融
版本）、模板编号、冻结哈希及 HTML；不读取或重采 records，不处理图片。
保留已有模板分配，临时文件在发布后清理。旧题面的完整 response 保留其原始输入哈希；
评估新版题面需要重新推理，使用独立输出文件。

输出放在 `data/benchmark/inference/responses/<alias>_response.json`，包含配置、
模型/图像预处理信息和 `predictions`。每条结果仅保存 `split`、`id`、`dataset`、
`task_id`、原始 `answer`、`finish_reason`（`stop` 或 `length`；SpatioLM 仅返回文本，记作 `unknown`）。
JSON 由脚本组装，不要求 VLM 生成 ID 或 JSON 外壳。
默认贪心生成，最多 4,096 个新 token，可用 `--max-new-tokens` 调整。
thinking 模式与小样本运行分别命名；只记录请求的模板开关，不改 benchmark 的 prompt。

每批更新同一份 JSON，再次运行相同命令按 `(split, id)` 续跑。
更换源 QA 或生成设置时，直接调用对应的 `infer_*.py --model ... --output ...` 指定其他结果路径，
不要多个进程同时写同一路径。
不复制图片、不修改 GT/records、不新增评分或 LLM 裁判逻辑。
离线运行时直接向对应脚本的 `--model` 传本地 snapshot 目录，避免 Transformers 4.57.3
在使用已缓存的模型名称时仍请求 tokenizer 元信息。

## SFT：只有训练数据

新版 `seen_v2` 从当前 Seen records 完整重建，覆盖 A1/A2/A3/A4/B1/B2/B3/C1。
共 **270,880 道 QA、53,553 条 records，其中 B3 为 36,709 道**。
排除清单直接读取冻结 benchmark 的 7,000 条 record UID，Unseen 不参与训练。
完整数量、覆盖率、任务分布和坏图路径以
`data/sft/seen_v2/metadata/summary.json` 为准。A2 在距离 rank 均衡后供给较少，
不重复稀缺题来凑成跨任务完全等量。跳过之前已知的 1 条 GS 原图损坏记录，
与冻结 benchmark 的 record 交集为 0。新版验收通过，旧 `seen_v1` 已删除。

极暗图指原分辨率下至少 90% 的像素，其 RGB 三通道均不超过 8。
已删除相关的 39 道 QA 和失去引用的 96 个导出图片文件，同步五份 JSONL
及 A3 标准类别名称；原始 records 和资产未改动。损失仅占 QA 的 0.0144%，
无需补选。精简清理记录在 `metadata/quality_cleanup.json`。
已有导出可运行 `python scripts/build_seen_benchmark.py clean-sft`；
后续 `compile-sft` 已自动跳过此类极暗初始图和 C1 选项图。

`compile-sft` 整条排除 benchmark 的 records，先覆盖剩余每条有有效 QA 的
record，再补较少的任务；每个 record/task 最多两道不同 QA。不是把所有表面点
全部展开，也不靠换模板重复同一道题。只支持部分任务的 record 同样保留。
任务目标由最少的可选题型容量确定，不固定总量。A1 在起手类型内平衡碰撞/安全；
A2 在 dataset/length/start 内平衡距离 rank。真实供给不足会报告，不使整轮构建失败。
长度、起手、类别、24 方向按实际供给轮换，不照搬 benchmark 的硬配额。
B3 复用 benchmark 的固定表面点、无碰撞序列和中途距离计算，GS 不参与 B1/B3。
B3 按距离区间、动作长度、起手、Forward 序号及 25%/50%/75% 检查点轮换；
不同检查点算不同问题，但仍受每 record/task 最多两题的限制。

```text
data/sft/seen_v2/
  images/{b1k,gs,r2r}/       共用原图、标记点图、C1 终点图
  json/full.jsonl           messages + images 格式的独立训练 QA
  metadata/selection.jsonl  固定选样、record/case/point、模板、输入与 GT
  metadata/summary.json     来源绑定、覆盖率和实际分布
```

原图、终点图优先硬链接，同一个 record/point 的标记图只画一次。
不跑 DINO，不新增 oracle/gate，不重采，不改原 records 或 benchmark；
不划分训练/验证/测试集，也不启动训练。
损坏的初始图只跳过该 record；损坏的终点图只影响对应 C1，能用的其他题仍保留。
具体文件和实际覆盖数量记在 `metadata/summary.json`，不修改或重采源数据。

参数消融直接从已保存的 `json/full.jsonl` 按需导出，不重新渲染模板、不复制图片、不重新选题：

```bash
$HABITAT_PY scripts/build_seen_benchmark.py export-sft --root data/sft/seen_v2 --hide-params height
$HABITAT_PY scripts/build_seen_benchmark.py export-sft --root data/sft/seen_v2 --hide-params radius
$HABITAT_PY scripts/build_seen_benchmark.py export-sft --root data/sft/seen_v2 --hide-params fov
$HABITAT_PY scripts/build_seen_benchmark.py export-sft --root data/sft/seen_v2 --hide-params height radius fov
```

对应 `json/no_height.jsonl`、`no_radius.jsonl`、`no_fov.jsonl`、
`no_parameters.jsonl`，也支持其他参数组合。各版本 ID、顺序、图片、模板、
C1 选项顺序、GT 完全一致。仅删除对应配置行；`no_parameters` 删除整个
`Configuration` 段。FOV 同时隐藏 HFOV/VFOV；system prompt、问题正文、动作
距离、转角和 A4/B3 的检查点百分比均不变。所有 SFT 版本的 `images` 均保存相对
数据集根目录的 `images/...`，不写机器绝对路径；整体移动 `images/`、`json/`、
`metadata/` 后不用重写 JSON。Benchmark 图像路径则相对于 `qa.json` 所在目录。

采用 [ms-swift 官方多模态格式](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html)，
A1–B3 一张图，C1 按顺序五张图。两个 SFT 启动脚本均用原生 `ROOT_IMAGE_DIR`
指定 SFT 根目录，解析 `images/...`，不重写 JSON、不注册自定义数据集。
直接使用保存的 system/user 和 assistant GT；SFT loss 只监督 assistant 回答。
编译数据不会启动训练，测试只用独立 benchmark。

### Qwen3-VL-4B 全参数 SFT（默认四张 A100）

`scripts/train_sft_full.sh` 同时训练语言模型、视觉编码器和连接模块，不使用
LoRA、不冻结这三部分。复用下文的 ms-swift 环境及 DeepSpeed（本机已装 0.18.3）。
在其他服务器上设置 `SFT_PYTHON` 指向兼容的 Python 环境；`SFT_MODEL` 可指定
本地基础模型目录。默认离线加载，尚未下载权重时设置 `HF_HUB_OFFLINE=0`。

| 配置 | 默认值与考虑 |
|---|---|
| GPU / 分片 | 四张 A100，BF16，原生 ZeRO-3，不做 CPU offload |
| 学习率 | 语言模型 `1e-5`、视觉编码器 `1e-6`、连接模块 `5e-6`；视觉侧小步更新，减少对预训练特征的扰动 |
| 有效 batch | `4 卡 × 每卡 1 条 × 累积 8 步 = 32` |
| 训练日程 | 一轮、cosine 衰减、3% warmup；这是初始配置，不是已调优的最优值 |
| 优化器 | AdamW，weight decay `0.01`，梯度裁剪 `1.0` |
| 图像 / 上下文 | 每图最多 1,024 个视觉 token，总长度 8,192；C1 保留完整五图 |
| 显存 / 数据 | 语言和视觉均启用梯度检查点，SDPA、按需处理图像；不 packing，不划验证集 |
| 保存 | 每 1,000 个优化器更新步保存，保留一个可续训 checkpoint |

使用 [ms-swift 3.12 原生参数](https://swift.readthedocs.io/en/v3.12/Instruction/Command-line-parameters.html)。
设置分模块学习率会自动启用其多模态优化器分组；日志中的 `learning_rate`
可能显示第一组（视觉侧）的值，而不是语言模型的值。不增加自定义 trainer、
数据适配器或 DeepSpeed JSON。

```bash
# 在四张 A100 的训练服务器启动，不是单卡机器。
bash scripts/train_sft_full.sh

# 共用图像；可选 full/no_height/no_radius/no_fov/no_parameters。
# 每个版本独立输出，并从同一基础模型开始。
SFT_VARIANT=no_height bash scripts/train_sft_full.sh

# 恢复优化器等状态，继续同一次训练。
bash scripts/train_sft_full.sh \
  --resume_from_checkpoint outputs/sft/qwen3vl-4b-full-finetune/full/checkpoint-1000
```

产物放在 `outputs/sft/qwen3vl-4b-full-finetune/<variant>/`，与历史 LoRA 输出隔离。
checkpoint 包含优化器状态，体积明显大于 LoRA；保存时先写新 checkpoint 再清旧的，
需预留临时双份空间。改变卡数或每卡 batch 时，要同步调整累积步数才能保持有效
batch 为 32。目前只做启动参数检查，未声称完成四卡 A100 的训练或显存实测。

### 用 ms-swift 训练 Qwen3-VL（LoRA 入口）

复用已有 `qwen3vl` Conda 环境，不修改 `qwen3vl_habitat`：

```bash
conda activate qwen3vl
python -m pip install --no-cache-dir ms-swift==3.12.6 \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 transformers==4.57.3 \
  peft==0.18.0 accelerate==1.12.0
bash scripts/train_sft_lora.sh
```

此环境固定 ms-swift 3.12.6：4.5.2 的训练回调无条件导入 PyTorch 2.5.1 没有的
`torch.distributed.fsdp.FSDPModule`。脚本使用 3.x 原生的 `--train_type lora`，
不修改第三方代码、不升级 PyTorch。
`full.jsonl` 表示保留全部机器人输入参数，不代表全参数微调。
为保持已有 checkpoint 路径不变，本次仅明确脚本名称，不迁移原输出目录。

默认使用本地已缓存的 Qwen3-VL-4B-Instruct，四卡 DDP、BF16 LoRA（rank 16、
alpha 32），冻结视觉编码器和连接模块，270,880 条 QA 训练一轮，有效 batch 为 32。
不划验证集、不在训练中评测 benchmark、不预计算视觉特征、不在启动前全量 packing；
图像按需加载。产物统一放在 `outputs/sft/qwen3vl-4b-full/`：最多两个 checkpoint
（保留优化器状态以便续训）、TensorBoard 曲线和原生训练元数据。

在项目根目录后台启动：

```bash
mkdir -p outputs/sft/qwen3vl-4b-full
nohup bash scripts/train_sft_lora.sh > outputs/sft/qwen3vl-4b-full/train.log 2>&1 < /dev/null &
```

可在脚本后追加 ms-swift 原生参数覆盖默认值，例如
`--resume_from_checkpoint outputs/sft/qwen3vl-4b-full/checkpoint-1000`。
参数消融先导出对应 JSONL，再同时更换 `--dataset` 和 `--output_dir`；每组从同一
基础模型开始，不接着 full 的结果训练。`SFT_PYTHON`、`SFT_MODEL` 可指定其他
解释器和模型；默认离线加载已有权重，下载其他模型时才设置 `HF_HUB_OFFLINE=0`。

## 任务定义

| 任务 | 问题 |
|---|---|
| A1 | 动作序列是否发生碰撞？ |
| A2 | 首次碰撞发生在哪个动作？ |
| A3 | 首次接触哪个初始可见物体/类别？ |
| A4 | Forward 内部检查点处，标记表面点在哪个方向？ |
| B1 | 终点相机到标记点的距离？ |
| B2 | 最终姿态下标记点的方向？ |
| B3 | 指定 Forward 完成 25%/50%/75% 时，相机光心到标记点的三维距离？ |
| C1 | 哪张图是真实终点视角？ |

GS 不支持 A3/B1。A4/B1/B2 和 C1 的 query/三个干扰项全部使用安全且完整执行的序列。
A3 当前沿用 records 中“首次接触类别”的既有定义，不将碰撞答案伪装成安全动作问题。
A2 保留交替动作导致的奇偶结构先验，评测分别报告两种起手的条件准确率。

## records 更新与新场景

Unseen 新场景使用 `scripts/collect_unseen_records.py`，直接调用现有采集器输出
`abc1.record.v3`，不构建 benchmark。R2R 使用官方 test 的 18 个场景，GS 使用
官方 InteriorGS val 的 9 个场景；B1K 使用项目预留的 10 个场景，**不是官方
test split**，其原始 catalog 的 split 仍为 train。

```bash
$HABITAT_PY scripts/collect_unseen_records.py \
  --data-root /home/zhangshan/syp/datasets \
  --b1k-source-manifest /home/zhangshan/syp/datasets/behavior-1k-v3.9.1/pbench-abc1-task4/catalog-authority-audit-v2-20260812/audit-derived-source-manifest.json \
  --b1k-python /home/zhangshan/miniconda3/envs/behavior/bin/python
```

默认 GPU 0/1 分别处理不重叠的 B1K 场景，GPU 2 处理 R2R，GPU 3 处理 GS。
每个场景加载一次并完成全部 pose 预算后才换场景。目标预算为 3000/3000/2000
条，均分场景后合计最多 8013 条；实际数量由有效产出决定。数据保存在
`data/metadata/test/unseen/<dataset>/<worker>/<scene>/`，收尾生成根目录
`manifest.json` 引用原始分片，不复制 records 或图像。`collection.json` 记录
进度；相同命令可续跑。Unseen 只用于评测，不混入 seen SFT。

已有 records 可通过 `plan/collect/dry-run`、`update-records`、`refine-surfaces`
更新。每个 worker 处理完当前场景全部 records 后才换场景，固定原物理参数；
单条 record 可以只支持部分任务，不设整条 record 必须集齐 ABC1 的门。
这些入口不因 benchmark 重构而删除。详见 [实现说明](BENCHMARK_IMPROVEMENTS.md)。
