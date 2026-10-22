# 模型推理

从当前 QA 生成模型原始回答，再交给[评测入口](../evaluation/README.md)评分。
旧 benchmark 的模型回答和评分已退役，不能作为当前 benchmark 的模型成绩。
以下命令均在项目根目录执行。

## 题集与运行目录

| 数据根目录 | 主集 | 半径诊断集 |
|---|---|---|
| `data/benchmark` | `seen`：5,000 题 | `seen_radius_pairs`：300 对、600 题 |
| `data/benchmark` | `unseen`：2,000 题 | `unseen_radius_pairs`：150 对、300 题 |

默认根目录是 `data/benchmark`，用 `--sets seen` 或 `--sets unseen` 选择分区；两个半径诊断集也使用同一根目录。
已完成的正式回答统一在 `data/benchmark/inference/responses/`；各次运行日志位于 `inference/runs/<model>/<set>/`。

## 先试跑，再全量

```bash
bash data/benchmark/inference/run.sh list

# 只打印命令，不启动模型。
bash data/benchmark/inference/run.sh qwen3vl-4b --dry-run --benchmark-root data/benchmark --sets seen

# 每任务一题，共 8 题，只检查推理链路。
bash data/benchmark/inference/run.sh qwen3vl-4b --cached --serve --answer-only --benchmark-root data/benchmark --sets seen --limit-per-task 1 --run-dir data/analysis/inference_smoke/qwen3vl-4b

# 完整 seen 主集。
bash data/benchmark/inference/run.sh qwen3vl-4b --cached --serve --answer-only --benchmark-root data/benchmark --sets seen --run-dir data/benchmark/inference/runs/qwen3vl-4b/seen
```

unseen 使用 `--sets unseen`；半径诊断使用对应的 `*_radius_pairs` 集名。
诊断必须包含一对的两题。`--limit-per-task` 的试跑回答带 `_smoke` 后缀，题目不全，评分入口会拒绝。
试跑后检查两种单位、未来画面题的五图顺序、空回答和截断；每任务一题的示例不保证覆盖这些情况，需要补查。

## 模型与输入设置

| 模型选择 | 范围 |
|---|---|
| 一个或多个 alias | 指定模型，名称和 checkpoint 见 `run.sh list` |
| `all` / `small` | 全部已注册基模 / 启动器中定义的小模型组 |

权重版本、输入版本、生成参数、thinking 模式和图像处理设置须在正式比较前确定。
默认贪心生成，最多 4,096 token；`--thinking auto/on/off` 控制模型支持的模式。
本轮使用 `--answer-only`：在每题末尾统一追加“只给最终答案，不输出解释、计算过程或推理”。
该指令及 thinking 设置写进 response JSON；支持 thinking 开关的模型使用 `--thinking off`。
原 QA 文件保持冻结，变更提示词后使用新的运行目录，从头生成，不能接着旧提示词的回答续跑。
各模型后端与默认设置见 [run.py](run.py)，启动器默认值不代表所有模型的官方配置。

`--variant` 支持 `full`、`no_radius`、`no_height`、`no_fov`、`no_parameters`。
默认使用完整 `QA.json`；训练权重名称不改变推理输入。
输入保留 System/User，不发送 assistant GT；未来画面题按初始图、A/B/C/D 顺序输入五张图片。
专用适配器负责各模型的图像处理和解码接口。

## 单参数消融

消融只针对主集：seen / unseen 各有独立的 `QA_no_radius.json`、`QA_no_height.json`、`QA_no_fov.json`。
半径诊断集只有完整配置的 `QA.json`（去掉半径后每对两题输入相同、答案相反）。
每份只删除指定配置行；去 FOV 同时删除水平和垂直视场角。其他提示词、题目 ID、图片、答案、单位完全相同。

本轮 27 个模型的权重、运行环境和完整参数版 response 哈希冻结在
`runs/ablations/manifest.json`。用同一入口重放，避免通用模型组包含旧的自训练 checkpoint：

```bash
bash data/benchmark/inference/run.sh qwen3vl-8b \
  --model-manifest data/benchmark/inference/runs/ablations/manifest.json \
  --variants no_radius no_height no_fov \
  --sets seen unseen \
  --run-dir data/benchmark/inference/runs/ablations/gpu1 \
  --gpus 1 --cached --background
```

回答仍写入 `inference/responses/`，文件名同时包含模型、题集和消融名；不会覆盖完整参数版。
每个文件完成后检查回答覆盖率，并与完整参数版核对解码、提示词和图像处理配置。
同一 API 模型的各题集、各消融共用一次服务加载。不同 GPU 使用不同运行目录和互不重叠的模型清单。
共用 GPU 时可用 `--gpu-memory-utilization` 降低缓存显存预算；该值取其与冻结配置的较小值，
实际预算记录在队列状态中。权重、精度、图像处理、上下文长度及解码参数仍保持一致。
半径对照结果单独统计；去半径后同一对输入相同、标签相反，整对零分是确定性回答下的必然结果。

## 输出与续跑

指定 `--run-dir` 后，回答在其 `responses/`，日志在 `logs/`，状态文件在该运行目录。
未指定时写入所选 benchmark 根目录的 `inference/`。
回答包含配置、QA 哈希、原始文本和 `finish_reason`，文件名区分题集、输入消融和试跑。
相同命令按 `(split, id)` 续跑，并检查输入与配置一致；更换设置使用新的运行目录。
同一回答文件只能有一个写入进程。

评分使用[评测入口](../evaluation/README.md)：按题集分组评分后自动合并。

## 环境

按 [.env.example](../../../.env.example) 配置 `.env.local`；不要把密钥写进运行记录。
`INFER_PYTHON` 用于启动器，模型环境由 `QWEN_PYTHON`、`MODERN_PYTHON`、`COSMOS_PYTHON`、
`CAMBRIAN_PYTHON`、`SPATIOLM_PYTHON`、`HY_PYTHON` 等变量选择。
`--cached` 使用本地权重并离线运行；`--serve` 在支持的模型上托管 vLLM，其他模型用原生适配器。
`--gpus` 选择设备，`--tensor-parallel-size` 设置服务并行度；已有服务可用 `--base-url URL`。
更多参数见 `bash data/benchmark/inference/run.sh --help`。
