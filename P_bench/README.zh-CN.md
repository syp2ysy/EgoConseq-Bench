# EgoConseq-Bench

[English README](README.md)

EgoConseq-Bench 从 R2R/Matterport3D、Habitat-GS/InteriorGS 和
BEHAVIOR-1K 三个模拟器采集第一视角、action-conditioned 的物理后果数据。
Pipeline 先保存模拟器真实计算得到的结构化 record，再确定性地编译成统一 QA。

本仓库包含采集、验证、QA 编译、评测和静态网页浏览器。仓库不再分发原始模拟器
资产；请按各数据源的许可从官方渠道下载。

## 快速开始

先克隆外层仓库并进入本 benchmark 子目录：

```bash
git clone https://github.com/syp2ysy/EgoConseq-Bench.git
cd EgoConseq-Bench/P_bench
export P_BENCH_ROOT="$PWD"
```

然后选择一条流程：

- **直接使用封存快照：** 按第 1 节创建 Habitat 环境，按“当前封存的训练快照”
  下载并解压 Hugging Face 的两个 archive，再运行第 5 节验证命令。之后可直接查看
  已编译 QA、保存的 replay 和静态网页，不会启动模拟器。
- **从三库重新采集：** 按第 1-2 节创建两个环境；把三套官方资产放进
  `data/sources/`（或设置对应覆盖变量）；按第 3 节生成 GS/B1K authority；先跑
  第 6 节三个 smoke，再按第 7 节启动基于 canary 的后台采集。

下载的数据必须真实解压到 `P_bench/data` 下，不要用软链接替换整个 `data/`；source
authority 会绑定字面路径。recovery snapshot 是不可变证据，不能用新代码 resume；
重新采集必须使用新的输出目录和 manifest。

## 实现了什么

当前 active contract 只有六个任务：

| ID | 给定初始第一视角图像和 action 程序后需要回答的问题 |
| --- | --- |
| A1 | 机器人是否会碰撞 |
| A2 | 第一次碰撞发生在第几个 action（1-based） |
| A3 | 第一次接触的是哪个初始可见物体或表面类别 |
| B1 | 安全执行后，选定的初始可见目标距离有多远 |
| B2 | 安全执行后，目标位于机器人前、左、右还是后方 |
| C1 | 四张真实终点渲染图中哪一张是真实未来视野 |

公开输入只有一张初始 RGB、body radius、相机光心高度、HFOV/VFOV、规范化
action 序列，以及 B 任务使用的目标。Depth、semantic、navmesh/完整几何、终点
位姿和 GT 证书均为私有信息。碰撞与安全标签要求完整几何 rollout、初始 depth
rollout 和 swept-corridor coverage 一致。

`A4_checkpoint_direction` 是从封存 record 编译的独立、非 headline 诊断轨：它
询问一个物体在某个 Forward action 的 25%、50% 或 75% 位置时相对机器人的方位。
它不会改变采集、六任务注册表或容量停止条件。

当前 candidate artifact 固定 `headline_eligible=false`。`--gt-as-pred` 只验证
scorer 完整性，不是模型成绩。

## 数据流和目录

```text
只读模拟器资产
  -> records/<dataset>/<shard>/<scene>/records.jsonl
  -> artifacts/global/<checkpoint>/candidate_qa/
  -> artifacts/global/<checkpoint>/candidate_qa_report/index.html
```

编译结果包括：

```text
candidate_qa/benchmark.json
candidate_qa/public/items.jsonl
candidate_qa/private/answers.jsonl
candidate_qa/private/atoms.jsonl
candidate_qa/private/source_map.json
candidate_qa/report.json
candidate_qa_report/index.html
```

record 转 QA 只读取封存 record 和已保存 PNG，不会重启模拟器。`pipeline/` 是唯一
benchmark 实现，`scripts/` 是 CLI，`tests/` 是确定性测试。

## 当前封存的训练快照

当前 recovery snapshot 是 train-seen、candidate-only 数据：

| 口径 | 数量 |
| --- | ---: |
| source shards | 562 |
| 模拟器 record | 10,028 |
| 精确唯一初始帧 | 9,798 |
| 六任务 QA | 44,404 |
| A1 / A2 / A3 | 3,772 / 5,989 / 5,691 |
| B1 / B2 / C1 | 7,551 / 10,531 / 10,870 |

根 checkpoint 为
`data/candidate_pool/abc1_train_seen_200k_20260816_451b61a/artifacts/global/recovery-562/checkpoint.json`，
SHA256 是
`968248000449cbaacde41871c0c30bec22d25464964cd8f41e12bfa5deb645b7`。

受限访问备份位于 Hugging Face dataset
`syp115/pbench-abc1-train-seen-recovery-562`。只有恢复这批历史快照才需要访问权；
从官方模拟器资产重新采集不依赖它。获权后执行：

```bash
huggingface-cli download syp115/pbench-abc1-train-seen-recovery-562 \
  --repo-type dataset --local-dir /path/to/recovery-562-download

cd /path/to/recovery-562-download
sha256sum -c SHA256SUMS
cat pbench-abc1-recovery-562.tar.zst.part-* | zstd -d | \
  tar -xf - -C "$P_BENCH_ROOT"
zstd -dc pbench-supporting-fixtures.tar.zst | \
  tar -xf - -C "$P_BENCH_ROOT"
```

解压目标必须是真实 clone 下来的 `P_bench` 目录，不能是软链接替身。使用前验证：

```bash
cd "$P_BENCH_ROOT"
PY="${EGOCONSEQ_HABITAT_PYTHON:-$(command -v python)}"
"$PY" scripts/check_abc_golden.py
echo "968248000449cbaacde41871c0c30bec22d25464964cd8f41e12bfa5deb645b7  data/candidate_pool/abc1_train_seen_200k_20260816_451b61a/artifacts/global/recovery-562/checkpoint.json" | \
  sha256sum -c -
"$PY" -m http.server 8789 --directory "$P_BENCH_ROOT"
# 打开 data/candidate_pool/abc1_train_seen_200k_20260816_451b61a/artifacts/global/recovery-562/candidate_qa_report/index.html
```

这批数据是可复算的训练/候选快照，并在 `docs/runs.json` 注册；不要把它描述成
completed 或 headline benchmark。

## 已测试环境

当前代码在 Linux、4 张 RTX 4090（每张 24 GB）、NVIDIA driver 575.57.08 和
CUDA 12.x user-space 包上验证。单场景 smoke 只需一张 GPU；后台 controller 可调度
多张 GPU。

必须使用两个 Python 环境，因为 Isaac Sim 5.1 和已验证的 Habitat stack 使用不同
Python 版本：

| Runtime | 已验证版本 |
| --- | --- |
| Habitat/R2R/GS 与 controller | Python 3.9.23、Habitat-Sim 0.2.4 headless、Habitat-Lab 0.2.420230405、PyTorch 2.5.1、gsplat 1.5.3 |
| BEHAVIOR-1K | Python 3.11.15、BEHAVIOR/OmniGibson 3.9.1、Isaac Sim 5.1.0、BDDL 3.7.0、PyTorch 2.7.0+cu128 |

## 1. 创建 Habitat 环境

```bash
cd "$P_BENCH_ROOT"

conda env create -f environment/habitat.yml
conda activate egoconseq-habitat
export EGOCONSEQ_HABITAT_PYTHON="$(command -v python)"

python - <<'PY'
import habitat_sim, habitat, gsplat, numpy, torch
print("Habitat-Sim", habitat_sim.__version__)
print("NumPy", numpy.__version__)
print("PyTorch", torch.__version__, "CUDA", torch.cuda.is_available())
PY
```

`environment/habitat.yml` 对齐当前已测试版本。PyTorch 使用 CUDA 12 wheel，请保证
驱动兼容。GS collision-authority 预处理读取官方 USD 时还需要：

```bash
python -m pip install usd-core
```

## 2. 安装 BEHAVIOR-1K / OmniGibson

在单独环境运行官方 v3.9.1 安装：

```bash
git clone -b v3.9.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval
```

已验证 checkout 为 tag `v3.9.1`、commit
`26f2c7ef7b9cf96bd0414f81e1e751e493762779`。机器相关的 Isaac Sim 安装以官方
说明为准：

- BEHAVIOR：<https://behavior.stanford.edu/getting_started/installation.html>
- Isaac Sim 5.1 Python：<https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html>

通过 `EGOCONSEQ_B1K_PYTHON` 指向安装后的 Python；代码不依赖本机硬编码路径。

## 3. 下载并整理模拟器资产

### R2R / Matterport3D

1. 从 <https://github.com/jacobkrantz/VLN-CE> 下载 R2R-VLNCE v1-3 episodes；
2. 在 <https://matterport.com/partners/meta> 申请 Matterport3D 学术访问并下载
   MP3D scans；
3. 在同一 MP3D 根目录中保留每个 scan 的 `.glb`、`.house`、`.navmesh` 和
   semantic PLY，并在根目录放置
   `mp3d_annotated_basis.scene_dataset_config.json`。

R2R episode JSON 在本项目中只提供 scene whitelist，instruction 和 navigation goal
不会成为 benchmark 输入。

### Habitat-GS / InteriorGS

下载以下官方资产：

- Habitat-GS：<https://huggingface.co/datasets/RukawaY/gs_scenes>，revision
  `034f5938c40c55b873da81b1b6717484b40faae9`；
- InteriorGS labels：<https://huggingface.co/datasets/spatialverse/InteriorGS>，
  revision `5201ed9fd11fc2b8ac23e069796c386dbbf8f943`；
- SAGE-3D collision mesh：
  <https://huggingface.co/datasets/spatialverse/SAGE-3D_Collision_Mesh>。

统一为：

```text
GS_ROOT/
  train/<scene>/scene.gs.ply
  train/<scene>/scene.navmesh
  train/<scene>/labels.json
  splits/train.json
```

`splits/train.json` 的 schema 为 `egoconseq.scene_manifest.v1`；每行记录
`scene_id`、相对 `path`、`source_scene` 和 `split: "train"`。随后一次性构建真实
collision authority：

```bash
"$EGOCONSEQ_HABITAT_PYTHON" scripts/build_gs_collision_authority.py \
  --data-root "$EGOCONSEQ_GS_ROOT" \
  --source-manifest "$EGOCONSEQ_GS_TRAIN_MANIFEST" \
  --collision-root "$SAGE3D_COLLISION_ROOT" --jobs 4 \
  --exclude-scene interior_0505_839970
```

该场景是已知源转换失败；不要静默丢弃其他场景。Gaussian ellipsoid 只负责渲染，
不能作为碰撞 GT。

### BEHAVIOR-1K authority

官方安装下载 dataset 后，绑定安装文件并派生逐场景 authority：

```bash
PY="$EGOCONSEQ_HABITAT_PYTHON"
mkdir -p "$P_BENCH_ROOT/data/b1k-authority/fragments"

"$PY" scripts/build_b1k_source_manifest.py verify-install \
  --data-root "$B1K_DATA_ROOT" \
  --source-root /path/to/BEHAVIOR-1K \
  --output "$P_BENCH_ROOT/data/b1k-authority/install.json"

"$PY" scripts/build_b1k_source_manifest.py derive-shard \
  --data-root "$B1K_DATA_ROOT" \
  --output-dir "$P_BENCH_ROOT/data/b1k-authority/fragments" \
  --scene-timeout-s 1200 --resume \
  --scenes <SCENE_ID_1> <SCENE_ID_2>

INSTALL_SHA=$(sha256sum "$P_BENCH_ROOT/data/b1k-authority/install.json" | cut -d' ' -f1)
"$PY" scripts/build_b1k_source_manifest.py assemble-catalog-audit \
  --data-root "$B1K_DATA_ROOT" \
  --install-manifest "$P_BENCH_ROOT/data/b1k-authority/install.json" \
  --expected-install-manifest-sha256 "$INSTALL_SHA" \
  --fragment-dir "$P_BENCH_ROOT/data/b1k-authority/fragments" \
  --output "$B1K_SOURCE_MANIFEST" \
  --audit-output "$B1K_CATALOG_AUDIT"
```

完整目录分片和 resume 参数见
`build_b1k_source_manifest.py <subcommand> --help`。派生过程自动调用
`EGOCONSEQ_B1K_PYTHON`。

## 4. 配置本机路径

```bash
cp .env.example .env.local
# 修改路径后：
source .env.local
PY="$EGOCONSEQ_HABITAT_PYTHON"
cd "$P_BENCH_ROOT"
```

`.env.local` 和 `data/` 已被 Git 忽略。原始数据应保持只读。正式采集前 Git 工作树
`EGOCONSEQ_DATA_ROOT` 默认是 `$P_BENCH_ROOT/data/sources`；只有资产布局不同时才需
用 R2R、Matterport3D、GS 的专用变量覆盖。Controller 中的 `B1K_DATA_ROOT` 同理
默认指向 `data/sources/behavior-1k-v3.9.1`。
必须干净，因为 run 会记录并强制检查完整 commit SHA。

## 5. 验证安装

```bash
"$PY" -m pytest -q
"$PY" scripts/check_abc_golden.py
```

Golden 会从三库紧凑 fixture 重建 QA、检查六任务覆盖并执行 GT-as-pred replay；它不会
启动模拟器，也不能代替真实采集吞吐 canary。

## 6. 每个后端跑一个 scene smoke

示例场景必须存在于本机已认证的 train-seen catalog。下列命令只缩小采集规模，不放松
科学门。

R2R：

```bash
REV=$(git rev-parse HEAD)
"$PY" scripts/collect.py --backend r2r --benchmark-partition train_seen \
  --scenes uNb9QFRL6hY --poses-per-scene 1 \
  --pose-candidates-per-scene 800 --ordinary-actions-per-pose 24 \
  --record-idle-stop-s 120 --scene-wallclock-stop-s 600 \
  --code-revision "$REV" --out data/smoke/r2r
```

GS：

```bash
"$PY" scripts/collect.py --backend gs --benchmark-partition train_seen \
  --scenes interior_0123_840023 --poses-per-scene 2 \
  --pose-candidates-per-scene 800 --ordinary-actions-per-pose 24 \
  --record-idle-stop-s 120 --scene-wallclock-stop-s 600 \
  --code-revision "$REV" --out data/smoke/gs
```

B1K 用 supervisor 隔离 Isaac Sim worker：

```bash
"$PY" scripts/run_b1k_collection_shard.py run \
  --data-root "$B1K_DATA_ROOT" --source-manifest "$B1K_SOURCE_MANIFEST" \
  --output-dir data/smoke/b1k --gpu-id 0 --shard-id smoke-b1k \
  --code-revision "$REV" --scene-timeout-s 1200 \
  --scenes Pomaria_0_garden --collect-args \
  --poses-per-scene 1 --pose-candidates-per-scene 6000 \
  --ordinary-actions-per-pose 24 --record-idle-stop-s 120 \
  --scene-wallclock-stop-s 900 --benchmark-partition train_seen
```

一帧只要支持至少一个 active task 就可以保存；不要求同一帧同时满足 A1/A2/A3/B1/B2/C1。

## 7. 多 GPU 后台采集

Controller 用真实 canary 计算容量，不从经验值硬猜。顺序固定为：构建 canary → 运行
canary → 生成 profile → 构建不可变 production manifest → 启动。

```bash
CANARY="$EGOCONSEQ_RUN_ROOT/my-canary"
RUN="$EGOCONSEQ_RUN_ROOT/my-train-seen-run"

"$PY" scripts/run_background_collection.py canary-build \
  --output-root "$CANARY" \
  --r2r-scenes uNb9QFRL6hY XcA2TqTSSAj \
  --gs-scenes interior_0123_840023 interior_0045_839925 \
  --b1k-scenes Pomaria_0_garden Pomaria_0_int \
  --python "$PY" --b1k-data-root "$B1K_DATA_ROOT" \
  --b1k-source-manifest "$B1K_SOURCE_MANIFEST" \
  --ordinary-actions-per-pose 24

"$PY" scripts/run_background_collection.py canary-run \
  --manifest "$CANARY/controller/canary-manifest.json"

"$PY" scripts/run_background_collection.py profile \
  --measurements "$CANARY/capacity-evidence.json" \
  --out "$CANARY/capacity-profile.json"

PROFILE_SHA=$(sha256sum "$CANARY/capacity-profile.json" | cut -d' ' -f1)
AUDIT_SHA=$(sha256sum "$B1K_CATALOG_AUDIT" | cut -d' ' -f1)
"$PY" scripts/run_background_collection.py build \
  --output-root "$RUN" --python "$PY" \
  --b1k-data-root "$B1K_DATA_ROOT" \
  --b1k-source-manifest "$B1K_SOURCE_MANIFEST" \
  --b1k-catalog-audit "$B1K_CATALOG_AUDIT" \
  --b1k-catalog-audit-sha256 "$AUDIT_SHA" \
  --capacity-profile "$CANARY/capacity-profile.json" \
  --capacity-profile-sha256 "$PROFILE_SHA" --rounds 20

nohup "$PY" scripts/run_background_collection.py run \
  --manifest "$RUN/controller/manifest.json" \
  > "$RUN/controller/nohup.log" 2>&1 &

"$PY" scripts/run_background_collection.py status \
  --manifest "$RUN/controller/manifest.json"
```

统一通过 controller 停止，不要直接杀 worker：

```bash
"$PY" scripts/run_background_collection.py stop \
  --manifest "$RUN/controller/manifest.json"
```

Controller 只调度 train-seen 场景，在 catalog pass 间累积 1.5 m / 45° pose
exclusions，并发布全局 QA checkpoint。单库目标是 floor，不会掩盖全局 capacity
shortfall。

## 8. 编译、查看和评测

后台 checkpoint 已包含 QA 和网页。若要给 run 增加独立 A4 诊断：

```bash
CHECKPOINT="$RUN/artifacts/global/<checkpoint>"
"$PY" scripts/build_checkpoint_direction.py \
  --run-root "$RUN" --output "$CHECKPOINT/candidate_qa/diagnostics/checkpoint_direction.v1" \
  --update-report "$CHECKPOINT/candidate_qa_report/index.html"
```

从仓库根目录启动静态服务：

```bash
"$PY" -m http.server 8789 --directory "$P_BENCH_ROOT"
# 浏览 http://127.0.0.1:8789/data/candidate_pool/<run>/artifacts/global/<checkpoint>/candidate_qa_report/index.html
```

评测必须显式提供外部 source-authority manifest；artifact 内的 private source map
不能自我授权：

```bash
BENCH="$RUN/artifacts/global/<checkpoint>/candidate_qa"
"$PY" scripts/eval_benchmark.py --benchmark "$BENCH" \
  --source-authority-manifest /path/to/source-authority.json \
  --gt-as-pred
```

真实模型使用 `--predictions predictions.jsonl` 替换 `--gt-as-pred`。

## 多样性和数据切分

- 训练只从冻结的 `train_seen` scene list 采集；
- 未来 benchmark 从完全独立的 `test_unseen` scene family 采集；
- split 单位是 scene family，不是单帧；
- 跨 pass pose exclusions 防止近重复相机位姿；
- compiler 以精确 source-frame SHA 为图像身份，执行确定性的 per-frame/per-task
  diversity selection；
- action shortlist 在物理认证前覆盖 L1-L6 与动态距离 strata；
- 某个任务不满足只会 withhold 该任务，不会要求一张有效图同时凑齐六个 head。

模块职责见 `pipeline/README.md`，保留产物的 provenance 见 `docs/runs.json`。

## 常见问题

- **B1K 启动了错误 Python：** 将 `EGOCONSEQ_B1K_PYTHON` 指到官方
  BEHAVIOR 环境的 Python。
- **找不到 scene：** 检查冻结 partition 和认证 source manifest，不要通过 glob
  临时加入未注册 scene。
- **resume 被拒：** resume 只允许相同 commit、相同 contract；代码或合同改变后应
  新建输出目录。
- **没有 B target 或 C1 neighbour 不足：** 这是逐任务 typed withhold，该帧仍可发布
  其他任务。
- **artifact 是 non-headline：** 当前 candidate protocol 本来如此。

## 许可

当前 checkout 没有声明项目级统一 license。模拟器和资产分别受 Habitat、
Matterport3D、Habitat-GS、InteriorGS、SAGE-3D、Isaac Sim、OmniGibson、BDDL 与
BEHAVIOR-1K 的许可和访问条款约束。不要通过本仓库再分发受限原始资产。
