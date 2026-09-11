# Benchmark 文本推理

每个模型系列有独立的 `infer_*.py`，同系列不同尺寸共用该系列文件。
模型加载和生成代码放在各自文件中；`common.py` 只共用 QA 输入、HTTP 通信、结果保存和续跑。
API 模型的客户端入口也单独放置，实际模型加载由各自的官方服务负责。
`run.py` 按模型选择脚本和环境，`run.sh` 是共用启动入口。

五个已缓存小模型的完整队列：

```bash
bash data/benchmark/inference/run.sh small --cached --serve --evaluate
```

依次推理 Qwen3-VL-4B、RoboBrain2.5-4B、RoboInter-3B、RynnBrain1.1-2B、Cosmos3-Edge，
每个模型覆盖 Seen 4,999 + Unseen 2,000 题、全部八任务。Qwen3-VL、RoboBrain、RynnBrain 使用独立 vLLM 服务；
RoboInter 使用其原生 Transformers 适配器，Cosmos3 使用原生 Reasoner。每个模型结束后释放 GPU，再处理下一个。
推理结束后仅加载一次 Qwen3-8B，逐个评估已完成的 response。
`--cached` 固定本地 snapshot 版本并启用离线模式，不下载模型。
RynnBrain 与 Cosmos3 默认关闭 thinking，其余保持模型模板默认值；可显式覆盖。
API 默认 batch/workers 32，原生推理 batch 上限 4。生成上限 4,096 token，保留截断标记。

`pipeline_status.json` 记录每个模型的 inference/evaluation 状态和输入哈希；
`logs/<alias>_infer.log`、`logs/<alias>_server.log`、`logs/<alias>_eval.log` 保存各阶段日志。
只有全部请求的模型推理与评分均完成，队列才标记 complete；失败会记录并返回非零退出码。
相同命令支持断点续跑，完整结果不会重新生成。队列对所选 GPU 加进程锁，防止重复启动。
正式 QA、原图、GT、records 和 SFT 均不改动。

```text
inference/
  infer_qwen25vl.py
  infer_qwen3vl.py
  infer_qwen35.py
  infer_rynnbrain.py
  infer_robobrain.py
  infer_robointer.py
  infer_cosmos_reason2.py
  infer_cosmos3.py
  infer_embodied_r15.py
  infer_spatiolm.py
  infer_sensenova.py
  infer_hy_embodied.py
  common.py
  run.py
  run.sh
  README.md
  responses/
  logs/
```

```bash
# 查看全部 15 个模型配置
bash data/benchmark/inference/run.sh list

# 只查看命令：不下载、不加载模型、不生成 response
bash data/benchmark/inference/run.sh all --dry-run

# 单模型，覆盖 Seen/Unseen 八个任务共 16 道题，包括 C1 五图
bash data/benchmark/inference/run.sh qwen3vl-4b --limit-per-task 1

# 指定多个模型，或使用 all 顺序跑全量
bash data/benchmark/inference/run.sh rynnbrain11-2b robobrain25-4b
bash data/benchmark/inference/run.sh all
```

RynnBrain1.1-2B 按官方 Transformers 示例关闭 thinking；RoboInter-VLM-3B
保留其默认模板。两个具身模型的全量命令为：

```bash
bash data/benchmark/inference/run.sh rynnbrain11-2b --thinking off --batch-size 4
bash data/benchmark/inference/run.sh robointer-3b --batch-size 4
```

分别保存为 `responses/rynnbrain11-2b_thinking_off_response.json` 和
`responses/robointer-3b_response.json`。RynnBrain 已有完整缓存时可加 `HF_HUB_OFFLINE=1`。
RoboInter 使用 `--cached` 解析本地 snapshot 后可完全离线加载。

也可以用对应模型的 Python 环境直接运行独立脚本，例如：

```bash
/home/zhangshan/miniconda3/envs/qwen3vl/bin/python -B \
  data/benchmark/inference/infer_qwen3vl.py \
  --model Qwen/Qwen3-VL-4B-Instruct --limit-per-task 1
```

独立脚本默认文件名使用 checkpoint 名，启动器使用模型 alias；
需要二者续跑同一结果时，用 `--output` 指向启动器生成的文件。

全量默认读取 Seen + Unseen 的全部 QA（当前 6,999 道）。每个模型单独保存到
`responses/<alias>_response.json`，重复执行按 `(split, id)` 续跑；已完成的模型不再加载。
不同参数版本、thinking 模式和小样本运行使用不同文件名。不要让两个进程同时写同一个结果。
模型文件由 Hugging Face 共用缓存，不复制到 benchmark；首次运行可能下载权重。

`all` 逐个结束模型进程；API 模型会自动启动专用本地服务，结束后关闭本次创建的服务。
服务使用自动分配的本地端口，不占用 HTML 的 8000。失败模型会明确列出并返回非零退出码，
其余模型继续运行；不会把接口检查或假回答写成推理结果。服务日志放在 `logs/<alias>_server.log`。

## 参数消融

Seen 和 Unseen 各自保存五份对齐的 QA，共用同一个 `images/`：

| `--variant` | 文件 | 删除的信息 |
|---|---|---|
| `full`（默认） | `QA.json` | 无 |
| `no_radius` | `QA_no_radius.json` | Collision footprint radius 配置行 |
| `no_height` | `QA_no_height.json` | Camera optical-center height 配置行 |
| `no_fov` | `QA_no_fov.json` | Horizontal / Vertical field of view 两行 |
| `no_parameters` | `QA_no_parameters.json` | 整段 Configuration |

与 SFT 一样，消融从已保存的完整版删行，不重新套模板；system prompt、动作、
问题、GT、ID、顺序和相对图片路径均保持一致。每个版本仍为 4,999 Seen + 2,000 Unseen，
不是新增题目。HTML 继续展示完整版。

```bash
# GPU/模型环境恢复后，推理指定版本（支持所有模型入口）
bash data/benchmark/inference/run.sh qwen3vl-4b --variant no_height
bash data/benchmark/inference/run.sh all --variant no_parameters
```

例如第一条保存到 `responses/qwen3vl-4b_no_height_response.json`，不混入 full 结果。
独立 `infer_*.py` 也支持 `--variant`；输入文件路径、摘要和版本保存在 response 配置中。

若以后经授权更新 `QA.json`，重新运行以下命令同步四份消融文件：

```bash
/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python -B \
  scripts/build_seen_benchmark.py export-benchmark --root data/benchmark
```

该命令只覆盖派生的 `QA_no_*.json`，不修改完整版、冻结索引、SFT 或图片。

## 环境

| 环境变量 | 使用模型 | 本机默认／所需环境 |
|---|---|---|
| `QWEN_PYTHON` | Qwen2.5/3-VL、RoboBrain2.5、RoboInter-VLM、Cosmos-Reason2、Embodied-R1.5 | 现有 `qwen3vl` 环境，Transformers 4.57.3 |
| `MODERN_PYTHON` | Qwen3.5、RynnBrain1.1、SenseNova-SI | 现有 `vllm_serve` 环境，Transformers 5.6.2 / vLLM 0.19.1 |
| `SPATIOLM_PYTHON` | SpatioLM Understanding | 需安装官方 SpatioLM 与其 LMMs-Eval 依赖的解释器 |
| `COSMOS_PYTHON` | Cosmos3-Edge Reasoner | `~/.local/share/pbench/cosmos3-runtime/bin/python`，Transformers 5.17.0；复用 `vllm_serve` 的 PyTorch 2.10.0+cu128 |
| `HY_PYTHON` | HY-Embodied-VLM-1.0 | 需安装官方 HYV3VL vLLM 插件的解释器 |

`INFER_PYTHON` 只选择启动器的 Python，默认现有 `qwen3vl`；真正的模型进程使用上表环境。
SpatioLM 和 HY 尚无本机默认环境；Cosmos 使用独立的轻量环境，不修改 Qwen 环境。按官方说明准备后设置变量即可，
无需修改对应模型的推理脚本：

- [SpatioLM 官方安装及推理](https://github.com/xiaomi-research/spatio-lm#-inference)：
  使用 `InternVL3RChatModel` 与官方 `load_image`，保留空间模块，不能当普通 InternVL 加载。
  仅输入保存的 RGB 图像和问题，不加载深度教师、不额外提供深度或相机 GT。
- [Cosmos3-Edge 官方 Reasoner](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/README.md#transformers)：
  默认通过 `AutoModelForImageTextToText` 只加载 Reasoner，BF16 文本生成；不加载 VAE，
  不调用 Generator 或 Policy-DROID。仍可通过 `--base-url` 使用兼容的官方 vLLM 服务。
- [HY 官方服务插件](https://huggingface.co/tencent/Hy-Embodied-VLM-1.0#quick-start-with-vllm-recommended)：
  使用 VLM 文本输出，不使用 VLA。默认 TP=4；激活 3B 不等于总显存只需 3B，
  四张 24GB 卡是否足够还需实测，不能保证。

若已经用官方镜像或插件启动服务，可以跳过自动启动：

```bash
bash data/benchmark/inference/run.sh cosmos3-edge --base-url http://127.0.0.1:8001/v1
```

服务的模型 ID 需与 `run.sh list` 列出的 checkpoint 一致。该模式不会停止外部服务。
受许可限制的模型需先获得 Hugging Face 访问权限并在本机登录；不在代码或日志里存 token。

本机 Cosmos 环境的最小安装与全量运行命令：

```bash
uv venv --system-site-packages --python /home/zhangshan/miniconda3/envs/vllm_serve/bin/python \
  /home/zhangshan/.local/share/pbench/cosmos3-runtime
uv pip install --no-cache --python /home/zhangshan/.local/share/pbench/cosmos3-runtime/bin/python \
  'transformers==5.17.0' 'safetensors==0.8.0'

# 使用官方非 thinking 模板；system/user 和保存的图片不变。
HF_HUB_OFFLINE=1 bash data/benchmark/inference/run.sh cosmos3-edge --gpus 0 --thinking off --batch-size 4
```

`HF_HUB_OFFLINE=1` 适用于本机已有完整缓存；首次下载时不设置。
这次运行输出 `responses/cosmos3-edge_thinking_off_response.json`，与默认 thinking 结果分开。
每批保存并支持续跑，完成后模型进程退出、释放显存。
本机 RTX 4090 已通过单图、C1 五图和混合四题 batch 的真实推理验证，峰值显存约 5.85 GiB。
全量推理日志为 `logs/cosmos3-edge_infer.log`；完成状态以日志和 response 条数为准。

## 输入与生成设置

- System/User 原文来自保存的 QA；不传 assistant GT，不追加任务提示。
- C1 按题面位置输入初始图和 A/B/C/D 四图，不拼图、不换顺序。API 发送原图文件字节；
  模型内部仍使用自身图像预处理。SpatioLM 的动态 patches 是官方输入预处理，不落盘。
- 默认贪心生成，最多 4,096 个新 token；`--max-new-tokens` 可调整。
- `--thinking auto` 保留模型默认值；`on/off` 仅通过 `enable_thinking` 模板参数请求，
  不改保存的 prompt。不是每个模型都提供开关；SpatioLM 官方 chat 接口仅支持 `auto`。
  不支持开关的其他模型可能忽略该参数，不能据此声称它实际关闭了思考。
- 保存原始文本 `answer`，服务单独返回 reasoning 时另存 `reasoning`；不在推理阶段评分。
  `finish_reason` 保留停止／截断状态；SpatioLM 官方 chat 只返回文本，因此记为 `unknown`，不冒充 `stop`。
- `--gpus 0` 或 `--gpus 0,1` 选择设备。API 模型可用 `--tensor-parallel-size 2`；
  普通模型默认一张卡，HY 默认四张卡。`--max-model-len` 控制本次自动启动服务的上下文窗口。

当前正式输出统一位于 `responses/`，使用当前 v8 QA。
续跑核对题目集合、输入哈希、模型与生成配置、推理代码摘要及可获得的运行时权重版本。
API 会检查 `/models` 返回的实际服务模型；HTTP 请求并发执行但按输入顺序保存。
每批返回数量必须与题数一致，不能静默漏题。原始长回答不在推理阶段裁剪或改写。
SpatioLM、HY 的专用环境仍须准备后才能运行。

对已有 response 评分使用相邻的 `../evaluation/run_eval.sh`，不重新进行图像推理。
评测说明见项目 README 的“对已保存的 response 评分”部分。
