# EgoConseq

[English](README.md)

从初始第一视角 RGB 图像预测机器人动作后果。八项任务分三类：碰撞事件（A1 是否碰撞、A2 第几个动作、
A3 撞到什么）、标记点的空间关系（A4/B2 方向，B3/B1 中途/终点距离）和视觉后果（C1）。Seen benchmark
共 5,000 题，Unseen 共 2,000 题，另有独立的半径对照诊断集；SFT 约 30 万题，来自同一批 seen 场景。具体定义见[任务与评分](data/benchmark/evaluation/README.md)。

## 数据与环境

代码：[EgoConseq-Bench](https://github.com/syp2ysy/EgoConseq-Bench/tree/main/P_bench)。
数据：[Hugging Face](https://huggingface.co/datasets/syp115/EgoConseq-Bench)。
数据归档包含 QA、图片、SFT、源 records 和已保存的实验产物；模型权重、外部模拟器另行安装。

在全新 checkout 的项目根目录下载并恢复全部分卷：

```bash
hf download syp115/EgoConseq-Bench --repo-type dataset --local-dir ../egoconseq-data-download
(cd ../egoconseq-data-download && sha256sum -c SHA256SUMS)
cat ../egoconseq-data-download/data.tar.zst.part-* | zstd -d | tar -xf -
```

解压会重建 `data/` 并覆盖同路径文件。将 [.env.example](.env.example) 复制为 `.env.local`，
填写本机运行环境和源数据路径后执行 `source .env.local`。
采集使用 [environment/habitat.yml](environment/habitat.yml)（Python 3.9）；
推理与评测使用独立的 Python 3.10+ 模型环境。

| 路径 | 内容 |
|---|---|
| `data/{seen,unseen}_meta_gt3/` | Meta 数据：图片、各任务 GT 表、C1 未来视角图、目标打点图 |
| `data/benchmark/benchmark/{seen,seen_radius_pairs}/` | Seen QA JSON：主集含完整版及四种参数消融；半径诊断只有完整版 |
| `data/benchmark/benchmark/{unseen,unseen_radius_pairs}/` | Unseen QA JSON：主集含完整版及四种参数消融；半径诊断只有完整版 |
| `data/benchmark/metadata/{seen,unseen}/` | 选题清单、分布报告和预览图 |
| `data/benchmark/release.json` | 发布清单：评测集、图片根目录、输入与文件哈希 |
| `data/sft/seen/` | SFT JSONL（五个版本）及选题元数据 |
| `data/analysis/benchmark_review/` | 多列 case 浏览、完整 QA 与发布数据统计 |
| `weights/` | 指向 `/data/zhangshan/weights/` 的软链接，本地模型权重实际存放于该目录 |

QA 中的图片路径相对于各发布清单指定的 GT3 meta 根目录，图片不复制。SFT 排除 benchmark 与诊断集用过的观测，
以及它们 0.5 m 内的所有观测。

## 推理与评测

- [推理说明](data/benchmark/inference/README.md)：模型选择、环境和参数消融。对发布目录用
  `--benchmark-root data/benchmark --sets seen`（或 `seen_radius_pairs`）选择评测集；Unseen 发布用
  `--benchmark-root data/benchmark --sets unseen`（或 `unseen_radius_pairs`）。
- [评测说明](data/benchmark/evaluation/README.md)：评分已有 response 并导出对比表；
  半径对照集单独报告单题与成对准确率，不计入主集。

## 构建与训练

Meta 数据（observation 及所有任务的 GT，Seen 与 Unseen 相同）由
[scripts/collect_meta.sh](scripts/collect_meta.sh) 调用 [pipeline/meta](pipeline/meta) 采集；
benchmark、半径对照集和（Seen 的）SFT 再由 [post_QA/build](post_QA/build) 按 spec 生成：
[seen.json](post_QA/specs/seen.json) 或 [unseen.json](post_QA/specs/unseen.json)。
当前题库使用 GT3 与冻结的图片审核表，规则见 [BENCHMARK_IMPROVEMENTS.md](BENCHMARK_IMPROVEMENTS.md)。
Seen、Unseen 共用 `data/benchmark`；历史记录只放在归档目录。

```bash
"$EGOCONSEQ_HABITAT_PYTHON" -m post_QA.build --spec post_QA/specs/seen.json     # data/benchmark、data/sft/seen
"$EGOCONSEQ_HABITAT_PYTHON" -m post_QA.build --spec post_QA/specs/unseen.json   # data/benchmark
"$EGOCONSEQ_HABITAT_PYTHON" -m post_QA.build --spec post_QA/specs/seen.json --draft --sft-total 20000
```

训练脚本使用 ms-swift 3.12.x，默认四张 GPU，读取 `data/sft/seen/`（`SFT_ROOT`），图片来自 `data/seen_meta_gt3/`（`ROOT_IMAGE_DIR`）。
按训练机器配置 `SFT_PYTHON`，可用 `SFT_MODEL` 指定权重；默认离线加载。

```bash
bash scripts/train_sft_full.sh
# 或使用 LoRA
bash scripts/train_sft_lora.sh
```

全参数训练通过 `SFT_VARIANT` 选择 `full/no_height/no_radius/no_fov/no_parameters`。
默认超参数和输出位置见[全参数脚本](scripts/train_sft_full.sh)与 [LoRA 脚本](scripts/train_sft_lora.sh)。

当前题库的完整 QA 和统计可在[本地审阅页](http://localhost:8765/analysis/benchmark_review/)查看。
更换发布数据后的重算方法见[页面说明](scripts/review_ui/README.md)。已有发布目录不可覆盖；再次构建需使用新输出目录。
