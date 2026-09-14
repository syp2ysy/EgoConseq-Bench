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
  zero3.json                # DeepSpeed ZeRO-3 配置
vln_baseline/
  model.py                  # 原生 Qwen 加载、视觉冻结
  dataset.py                # 轨迹切块、历史采样、导航 prompt
  collator.py               # 多模态编码、padding、动作 labels
  actions.py                # 动作文本与基础动作之间的转换
  habitat_extensions.py     # R2R-CE 数据集和评估指标注册
  eval_metrics.py           # 结果汇总
tests/                     # 轻量回归测试，不参与训练
outputs/                    # 训练产生的权重，不进入 Git
results/                    # 评估产生的结果，不进入 Git
```

以下命令均在项目根目录运行。`outputs/`、`results/` 初始为空。

## 2. 训练环境

训练不需要安装 Habitat，也不需要 MP3D 场景文件，只读取已采集的 RGB 和轨迹标注。

### 硬件和系统

- Linux、NVIDIA GPU、可用的 NVIDIA 驱动。
- 默认使用 bf16 和 FlashAttention 2，需要支持相应计算的 GPU。
- 编译 FlashAttention/DeepSpeed 扩展需要 CUDA Toolkit、C++ 编译器和 Ninja。
- ZeRO-3 将优化器状态放到 CPU，需要充足的主机内存和磁盘空间。

默认配方来自四卡训练，每卡 batch 为 4。脚本也支持其他 GPU 数；全局有效 batch 会随之变化。
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

### 数据布局

数据来源：[cywan/StreamVLN-Trajectory-Data](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data)。
准备并解压所需的 R2R/RxR 图像，使目录符合：

```text
StreamVLN-Trajectory-Data/
  R2R/
    annotations_v1-3.json
    images/{trajectory}/rgb/000.jpg
    images/{trajectory}/rgb/001.jpg
    ...
  RxR/
    annotations.json
    images/{trajectory}/rgb/000.jpg
    ...
```

修改 `config/train_r2r.yaml` 或 `config/train_r2r_rxr.yaml` 中的两个字段：

```yaml
annotation_path:
  - /your/data/StreamVLN-Trajectory-Data/R2R/annotations_v1-3.json
image_root:
  - /your/data/StreamVLN-Trajectory-Data/R2R/images
```

R2R＋RxR 配置包含两组对应路径。当前仓库配置指向本机
`/home/zhangshan/syp/datasets/StreamVLN-Trajectory-Data/`。
混合训练直接拼接两个来源的样本，没有额外的数据集采样权重。

### dataloader 与监督方式

```text
轨迹标注 → 按动作块构造样本 → 历史 RGB＋当前 RGB＋指令
        → Qwen chat template / processor → batch＋assistant labels
```

- 历史图像均匀采样，最多 8 张；起点以当前帧填充历史图像位置。
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

脚本会自动激活 `qwen3vl`，默认使用所有可见 GPU。选择设备：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_r2r.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_r2r_rxr.sh
```

`NUM_GPUS` 可限制训练进程数；它不应超过可见 GPU 数。
Miniconda 默认安装在 `$HOME/miniconda3`，其他位置通过 `CONDA_ROOT` 指定：

```bash
CONDA_ROOT=/opt/miniconda3 CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_r2r.sh
```

主要训练设置由 YAML 控制：

| 设置 | 默认值 |
|---|---|
| 模型更新 | 冻结视觉编码器，训练原生 visual merger、语言模型和 LM head |
| 训练目标 | 动作 token 交叉熵 |
| Epoch / 学习率 | 1 / 2e-5，cosine 调度 |
| 每卡 batch / 梯度累积 | 4 / 4 |
| 全局有效 batch | GPU 数 × 每卡 batch × 梯度累积；四卡为 64 |
| 精度 / 分布式 | bf16 / DeepSpeed ZeRO-3，CPU optimizer offload |
| 保存方式 | 每个 epoch 保存 checkpoint，保留最近 2 个 |
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

自定义输出和断点恢复：

```bash
bash scripts/train_r2r.sh --output_dir outputs/my_r2r_run
# 将路径替换为实际存在、带优化器状态的 checkpoint 目录
bash scripts/train_r2r_rxr.sh --resume_from_checkpoint /path/to/checkpoint-N
```

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

## 6. 回归测试（可选）

`tests/` 包含当前 baseline 的轻量回归测试，不参与训练，
不会增加模型显存占用；测试使用临时样本和替代模型，不下载权重，不需要真实数据、GPU 或 Habitat。

| 测试文件 | 检查内容 |
|---|---|
| `test_actions.py` | 动作分块、文本解析、执行段数 |
| `test_dataset.py` | 历史采样、单轮样本、STOP 和随机分块 |
| `test_collator.py` | 只监督 assistant 动作答案 |
| `test_model.py` | Qwen Processor 加载参数 |
| `test_train.py` | 视觉冻结、训练配置和 ZeRO-3 检查 |
| `test_eval.py` | 图像历史、生成参数和评估入口 |
| `test_eval_metrics.py` | 指标聚合、分片去重与完整性 |

测试工具为可选安装，不列入训练必需依赖：

```bash
conda activate qwen3vl
python -m pip install pytest==8.4.2
python -m pytest -q
```

当前共 33 个测试。这些检查不能代替真实模型训练或完整导航评估。

## 7. 验证范围

清理时已验证：两个数据配置的样本数量及抽样 batch 与原版一致；原始 4B 权重的
确定性动作生成一致且可以反向传播；两个启动入口通过缩小版原生 Qwen3-VL 的 ZeRO-3
短训练与保存；真实 4B 模型完成一个 Habitat episode。

尚未重新运行完整训练及完整验证集。项目内不附带训练结果或 checkpoint；
`outputs/`、`results/` 由后续运行生成。
