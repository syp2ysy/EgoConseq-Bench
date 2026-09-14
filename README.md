# Qwen3-VL VLN Baseline

用原生 **Qwen3-VL-4B-Instruct** 完成视觉语言导航：输入导航指令、历史 RGB 和当前 RGB，
输出动作文本。训练只监督动作答案，采用 NAVIDA 风格的单轮样本组织。

项目提供两个训练入口：**R2R** 和 **R2R＋RxR**。它们共享模型、dataloader 和训练代码，
仅数据来源和输出目录不同。模型没有额外的 depth、geometry 或逆动力学辅助任务。

## 1. 项目结构

```text
train.py                    # Hugging Face Trainer 训练入口
eval.py                     # Habitat R2R 导航评估
requirements.txt            # 训练环境的主要依赖与版本
config/
  train_r2r.yaml             # R2R 训练配置
  train_r2r_rxr.yaml         # R2R＋RxR 训练配置
  vln_r2r.yaml               # Habitat RGB 环境、评估数据和场景路径
scripts/
  train_r2r.sh               # 启动 R2R 训练
  train_r2r_rxr.sh           # 启动 R2R＋RxR 训练
  eval_r2r.sh                # 完整 R2R 评估与结果聚合
  aggregate_eval_results.py  # episode 去重和分片完整性检查
  pack_trajectories.py       # 每轨迹一个 ZIP，逐帧校验、可重复运行
  zero2.json                # NAVIDA 官方 DeepSpeed ZeRO-2 配置
vln_baseline/
  model.py                  # 原生 Qwen 加载、视觉冻结
  dataset.py                # 轨迹切块、历史采样、导航 prompt
  collator.py               # 多模态编码、padding、动作 labels
  actions.py                # 动作文本与基础动作之间的转换
  habitat_extensions.py     # R2R-CE 数据集和评估指标注册
  eval_metrics.py           # 结果汇总
```

以下命令均在项目根目录运行。`outputs/`、`results/` 由训练和评估时自动生成。

## 2. 训练环境

训练不需要安装 Habitat，也不需要 MP3D 场景文件，只读取已采集的 RGB 和轨迹标注。

### 硬件和系统

- Linux、NVIDIA GPU、可用的 NVIDIA 驱动。
- 默认使用 bf16 和 FlashAttention 2，需要支持相应计算的 GPU。
- 编译 FlashAttention/DeepSpeed 扩展需要 CUDA Toolkit、C++ 编译器和 Ninja。
- 使用 NAVIDA 官方 ZeRO-2 配置，不启用 CPU optimizer offload；每张 GPU 保留完整模型参数，梯度和优化器状态分片。

默认对齐 NAVIDA 官方四卡 SFT 配方，每卡 batch=4、梯度累积=4，全局有效 batch=64。
单张 24 GB GPU 已验证 4B 模型的单样本前向/反向，但这不等于默认 batch=4 的完整训练已经验证。
显存不足时，先在 YAML 中降低 `per_device_train_batch_size`，再按需要调整梯度累积。

### 已验证的本机版本

| 组件 | 版本 |
|---|---|
| Conda 环境名 | `qwen3vl` |
| Python | 3.10.19 |
| PyTorch / torchvision | 2.5.1+cu121 / 0.20.1+cu121 |
| CUDA Toolkit | 12.1 |
| Transformers | 4.57.3 |
| Accelerate | 1.12.0 |
| DeepSpeed | 0.18.3 |
| FlashAttention | 2.8.3 |
| NumPy / Pillow | 2.1.2 / 11.3.0 |
| TensorBoard | 2.21.0 |

`requirements.txt` 固定主要依赖，避免直接安装最新版本改变训练行为。
它不是整个 Conda 环境的锁文件；下列安装顺序根据已验证版本整理，尚未在全新环境重建验证。
已有本机 `qwen3vl` 环境时，可直接使用，无需重装。

### 新建训练环境

先在系统中准备 CUDA Toolkit 12.1 和 C++ 编译工具，再执行：

```bash
conda create -n qwen3vl python=3.10 -y
conda activate qwen3vl

# 改成实际 CUDA Toolkit 安装目录；本机为此路径
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version

# 先安装 CUDA 版本的 PyTorch
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121

# 再安装项目依赖与 FlashAttention
python -m pip install -r requirements.txt
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
```

PyTorch 安装组合见[官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v251)。
FlashAttention 的编译依赖和安装方式见[官方说明](https://github.com/Dao-AILab/flash-attention#installation-and-features)。
使用 `nvidia-smi` 检查驱动，使用 `nvcc --version` 检查 Toolkit；PyTorch 自带 CUDA runtime 不代表系统已有编译器。

检查关键依赖：

```bash
python - <<'PY'
import torch, torchvision, transformers, deepspeed, flash_attn
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
print("BF16:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
print("Transformers:", transformers.__version__)
print("DeepSpeed:", deepspeed.__version__, "FlashAttention:", flash_attn.__version__)
PY
```

## 3. 模型和训练数据

### 基础模型

两个训练配置默认使用 `Qwen/Qwen3-VL-4B-Instruct`，首次加载需要下载权重和 processor。
离线训练时，将 YAML 的 `model_name` 改成本地模型目录，或在启动命令中传入
`--model_name /path/to/Qwen3-VL-4B-Instruct`。本地目录应包含权重、config、tokenizer 和 processor 文件。

### 数据布局与 ZIP 打包

数据来源：[cywan/StreamVLN-Trajectory-Data](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data)。
默认训练配置使用**每条轨迹一个 ZIP**：保留完整轨迹的原始 JPG，使用 `ZIP_STORED` 只打包、
不额外压缩。训练直接按帧名索引读取，不需要解压到磁盘，也不需要读取整条轨迹。

```text
/data/zhangshan/StreamVLN-Trajectory-Data-ZIP/
  packing_report.json
  R2R/
    annotations_v1-3.json
    images/{trajectory}.zip   # 包内 rgb/000.jpg、rgb/001.jpg、...
  RxR/
    annotations.json
    images/{trajectory}.zip   # 包内 rgb/000.jpg、rgb/001.jpg、...
```

本机完整 R2R＋RxR 包含 30,809 条轨迹、2,548,787 张 RGB。
打包后为 30,809 个 ZIP，另有两份标注和一份校验汇总，图像内容和帧数不变。

在已有解压图像的机器上运行：

```bash
python scripts/pack_trajectories.py \
  --source-root /home/zhangshan/syp/datasets/StreamVLN-Trajectory-Data \
  --output-root /data/zhangshan/StreamVLN-Trajectory-Data-ZIP \
  --workers 4
```

源目录应包含 `R2R/annotations_v1-3.json`、`RxR/annotations.json`，以及对应的
`R2R/images/{trajectory}/rgb/*.jpg` 和 `RxR/images/{trajectory}/rgb/*.jpg`。
只打包 R2R 时增加 `--datasets R2R`。脚本仅使用 Python 标准库。

每个 ZIP 的全部帧经 SHA-256 比对后才改为正式文件名，标注原样复制。
重复运行会校验已有 ZIP 并跳过重写；内容不匹配会报错，不会悄悄覆盖。
原始 JPG 保留。输出路径需要可写，并预留接近原始 JPEG 总大小的空间；打包主要减少文件数。

换机器或目录时，修改 `config/train_r2r.yaml` / `config/train_r2r_rxr.yaml`：

```yaml
annotation_path:
  - /your/data/StreamVLN-Trajectory-Data-ZIP/R2R/annotations_v1-3.json
image_root:
  - /your/data/StreamVLN-Trajectory-Data-ZIP/R2R/images
```

R2R＋RxR 配置包含两组对应路径。dataloader 也兼容原始的
`images/{trajectory}/rgb/*.jpg` 目录；同一路径下若存在对应 ZIP，则优先读取 ZIP。
也支持 Hugging Face 上传版的 `images/{scene}/{trajectory}.zip` 布局：使用随包标注，
`image_root` 仍指向 `R2R/images` 或 `RxR/images`，无需修改代码或解压 ZIP。
混合训练直接拼接两个来源的样本，没有额外的数据集采样权重。

### dataloader 与监督方式

```text
轨迹标注 → 按动作块构造样本 → 历史 RGB＋当前 RGB＋指令
        → Qwen chat template / processor → batch＋assistant labels
```

- 历史图像均匀采样，最多 8 张；起点以当前帧填充历史图像位置。
- ZIP 只改变图像读取方式：按原有文件名排序，一个样本打开一次 ZIP，读取选中的帧后关闭；不同 worker 不共享文件句柄。
- 图像先 resize 到 308×252（宽×高），再交给 Qwen Processor。
- 单轮 assistant 答案是动作文本，例如 `forward 75 cm, turn left 30 degree`。
- 随机动作分块最多 3 段，合并概率 0.7，每段最多合并 3 个同类基础动作。
- 分块在 Dataset 初始化时由 seed 41 固定；含 STOP 的样本两倍采样。
- 用户输入和 padding 的 labels 为 `-100`，仅训练 assistant 动作答案和结束标记。

当前数据和配置生成 R2R 152,081 个样本、R2R＋RxR 583,910 个样本，包含 STOP 重复采样。

## 4. 启动训练

```bash
# R2R
bash scripts/train_r2r.sh

# R2R＋RxR
bash scripts/train_r2r_rxr.sh
```

脚本会自动激活 `qwen3vl`，默认启动 4 个训练进程。通过 `CUDA_VISIBLE_DEVICES` 选择四张卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_r2r.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_r2r_rxr.sh
```

启动时检查可见 GPU 数，不足 4 张会直接报错。`MASTER_PORT` 默认 25420，可在端口冲突时修改。
`NUM_GPUS` 可覆盖进程数，但变更后全局有效 batch 也会变化，不再是默认四卡配方。
Miniconda 默认安装在 `$HOME/miniconda3`，其他位置通过 `CONDA_ROOT` 指定：

```bash
CONDA_ROOT=/opt/miniconda3 CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_r2r.sh
```

两个训练 YAML 的 SFT 参数对齐 [NAVIDA 官方 train.sh](https://github.com/waynechu1021/NAVIDA/blob/main/scripts/train.sh)，
ZeRO 配置对齐其 [zero2.json](https://github.com/waynechu1021/NAVIDA/blob/main/scripts/zero2.json)。
模型使用 Qwen3-VL-4B，数据使用本项目的 R2R 或 R2R＋RxR ZIP 轨迹。

主要训练设置由 YAML 控制：

| 设置 | 默认值 |
|---|---|
| 模型更新 | 冻结视觉编码器，训练原生 visual merger、语言模型和 LM head |
| 训练目标 | 动作 token 交叉熵 |
| Epoch / 学习率 | 1 / 2e-5，cosine 调度 |
| 每卡 batch / 梯度累积 | 4 / 4 |
| 全局有效 batch | GPU 数 × 每卡 batch × 梯度累积；四卡为 64 |
| 精度 / 注意力 | bf16 / FlashAttention 2 |
| 分布式 / 梯度检查点 | DeepSpeed ZeRO-2 / 开启 |
| DataLoader | 每进程 4 workers，pin_memory 开启 |
| 日志 | 每 5 个 optimizer steps，TensorBoard |
| 训练中评估 / 保存 | eval_strategy="no" / save_strategy="no" |
| eval_steps / save_steps | 100 / 12000；关闭对应策略时不触发 |
| 训练 seed | 42，与官方 TrainingArguments 默认值一致 |
| 优化器默认值 | AdamW，weight_decay=0、warmup=0、max_grad_norm=1 |
| 保存方式 | SFT 完成后保存最终模型和 processor |
| 最终权重目录 | `outputs/qwen3vl_r2r` 或 `outputs/qwen3vl_r2r_rxr` |

建议正式训练前，先检查两种配置能否读到图像并生成 batch：

```bash
conda activate qwen3vl
python train.py --config config/train_r2r.yaml --dry_run_batch_only
python train.py --config config/train_r2r_rxr.yaml --dry_run_batch_only
```

该检查只加载 processor 和数据，不加载模型权重，也不证明完整训练的显存足够。
小规模训练可复制 YAML 到项目外的临时路径，设置 `max_samples`、`max_steps` 和更小的 batch，
再用 `bash scripts/train_r2r.sh --config /path/to/smoke.yaml` 启动。

自定义输出目录：

```bash
bash scripts/train_r2r.sh --output_dir outputs/my_r2r_run
```

TensorBoard 日志位于训练输出目录的 `runs/` 下。默认配方只保存最终模型；
需要中途断点恢复时，可在 YAML 中启用 `save_strategy: steps`，再通过
`--resume_from_checkpoint /path/to/checkpoint-N` 恢复。

## 5. R2R 导航评估

### 评估环境与资产

评估使用独立的 `qwen3vl_habitat` 环境，训练不依赖它。当前可运行版本为：

| 组件 | 版本 |
|---|---|
| Python | 3.9.23 |
| Habitat-Sim | 0.2.4 |
| Habitat-Lab / Habitat-Baselines | 0.2.4 系列，本机包版本为 `0.2.420230405` |
| PyTorch / torchvision | 2.5.1 / 0.20.1 |
| Transformers / FlashAttention | 4.57.3 / 2.8.3 |
| NumPy | 1.26.4 |
| Hydra / OmegaConf | 1.3.2 / 2.3.0 |

新机器需按 [Habitat-Sim v0.2.4](https://github.com/facebookresearch/habitat-sim/tree/v0.2.4#installation)
和 [Habitat-Lab v0.2.4](https://github.com/facebookresearch/habitat-lab/tree/v0.2.4#installation)
安装对应版本及 `habitat-baselines`，服务器使用 headless/EGL 版本。
这是单独的环境：不要将面向 Python 3.10、NumPy 2 的训练 `requirements.txt` 覆盖进去。
本项目尚未提供从零重建评估环境的完整锁文件。

除了 Python 环境，还需要 R2R-CE episode 文件和 MP3D 场景网格。
`config/vln_r2r.yaml` 当前设置为：

```yaml
scenes_dir: /home/zhangshan/syp/datasets/scene_datasets/
data_path: /home/zhangshan/syp/datasets/r2r_vlnce_v1-3/{split}/{split}.json.gz
```

场景文件应位于 `scene_datasets/mp3d/{scene}/{scene}.glb`。
这两类文件与训练用的 StreamVLN RGB 轨迹不同。换机器时修改上述路径。

### 评估命令

训练完成后，两个模型都可以在 R2R 上评估：

```bash
bash scripts/eval_r2r.sh outputs/qwen3vl_r2r results/r2r_val_unseen val_unseen
bash scripts/eval_r2r.sh outputs/qwen3vl_r2r_rxr results/r2r_rxr_val_unseen val_unseen
```

脚本自动激活 `qwen3vl_habitat`，每张可见 GPU 运行一个不重叠分片，结束后聚合结果。
输出目录必须为空。完整 `val_unseen` 检查 1,839 个 episode，以及分片缺失和重复情况。
逐 episode 结果在 `split_*/log/`，最终指标在输出目录的 `final_metrics.json`。

只测试一个 episode 时直接调用 Python 入口：

```bash
conda activate qwen3vl_habitat
python eval.py --model_path outputs/qwen3vl_r2r \
  --output /tmp/qwen3vl_eval_check --max_episodes 1 --save_traces
```

`eval_r2r.sh` 专用于完整评估，不接受 `--max_episodes`。

### 动作执行和评估协议

- 基础动作：前进 25 cm、左转 15°、右转 15°、STOP。
- 每次执行生成文本的前 2 个动作段，展开成基础动作后再做下一次决策。
- 默认最多生成 64 个 token，采样解码，temperature 0.2、top_p 1.0、repetition_penalty 1.05。
- 历史缓存最多 200 张 RGB，每次从中采样；之前的 assistant 文本不回填到下一轮 prompt。
- 保留历史 baseline 的外部早停：超过 400 步，或 `distance_to_goal` 连续不变超过 25 次时强制 STOP。
  真实距离用于早停和指标计算，不输入 Qwen；改变该规则后需重新评估。

## 6. 验证范围

ZIP 存储已完成全量校验：2,548,787 张图片逐帧 SHA-256 一致，30,809 条轨迹的帧顺序一致；
两种训练配置的全部样本索引、动作答案和动作消耗步数一致，混合数据全部 583,910 个样本的历史采样索引一致。
真实样本及 Qwen batch 的图像张量、文本 token 和 labels 已对照原 JPG 读取结果验证。
ZIP 适配只改变存储读取；训练 seed=42，动作分块的独立 seed=41，历史均匀采样最多 8 帧。

本机 4-worker、batch=4 的 dataloader 测试（含 Qwen processor 和张量传输），随机 512 个样本，
预热后交替测量三次，中位吞吐为散装 JPG 51.95 样本/秒、ZIP 52.39 样本/秒。
这不包含模型前向/反向，也不代表其他磁盘环境的性能。两个配置均已验证可以从 ZIP 生成训练 batch。

尚未使用当前 ZeRO-2 配方完成四卡训练及完整导航评估。项目不附带测试脚本、临时验证数据或 checkpoint。
