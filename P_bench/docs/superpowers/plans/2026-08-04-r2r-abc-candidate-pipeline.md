# R2R Visible-Space ABC Candidate Pipeline

日期：2026-08-04  
状态：ABC 设计冻结，候选流水线已实现；D 暂停，待重新设计后另立计划。

## 1. 目标与边界

用 R2R-train 对应的 MP3D/Habitat 场景，从真实初始 RGB 和真实 simulator rollout 直接生成
六类候选 QA、精确 GT、完整 benchmark JSON 与静态检查网页。

- 只实现 `A1/A2/A3/B1/B2/C1`，不实现 D、旧 Q1–Q10、Local Affordance 或诊断题。
- 只测试初始可视空间内的后果；盲区碰撞与离开可认证走廊的路径不发布。
- 公开条件是 RGB、body radius、相机离地高度、HFOV/VFOV、随机 action 序列，以及 B 的目标。
- Depth、semantic、navmesh、endpoint、certificate 与 GT 保持 private。
- 复用现有 `balanced_action_pool` 和 collection 的随机 action 路径，不新增 action family/bank。
- Candidate artifact 固定 `headline_eligible=false`，用于逐题检查，不冒充正式 leaderboard。
- `run_meta` 外部 SHA 在候选阶段负责文件完整性；它不替代 `docs/runs.json` 的 completed/clean-code
  formal authority。当前真实展示可用于验题，不可称为正式 release。

唯一 taxonomy/GT 上游：
[`2026-08-03-abc-first-order-consequence-freeze-design.md`](../specs/2026-08-03-abc-first-order-consequence-freeze-design.md)。

## 2. 六个正式任务

| Head | Task ID | 问题 | 答案 | 精确 GT |
|---|---|---|---|---|
| A | `A1_collision` | 执行全部 actions 是否碰撞？ | `collision / no_collision` | body-conditioned navmesh rollout 与 initial-depth rollout 共识 |
| A | `A2_collision_step_grounding` | 已知会碰撞，发生在第几个 action？ | `action_1 ... action_N` | 首次接触弧长映射回原始 1-based action index；full/depth index 一致且远离边界 |
| A | `A3_contact_object` | 已知会碰撞，最先碰到哪类可见物体或表面？ | `chair / cabinet / wall / ...` | full/depth 都归因到同一个非空、初始可见 instance，再读取语义类别 |
| B | `B1_endpoint_distance` | 安全执行后，终点离目标多少米？ | 四个米制选项之一 | true endpoint 到目标完整 ground-band triangle support 的最近水平距离 |
| B | `B2_endpoint_direction` | 安全执行后，目标位于最终朝向何方？ | `front / left / right / rear` | true endpoint ego frame 下，目标完整 triangle reference centroid 的 bearing |
| C | `C1_future_view_selection` | 安全执行后，哪幅图是真实终点视野？ | 四个 image choice ID 之一 | true endpoint 原生无损 RGB；错误项来自匹配的 completed-clear siblings |

统一路由：

- `collision` rollout 可生成 A1、A2、A3。
- `completed_clear` rollout 可生成 A1、B1、B2、C1。
- 一个真实 rollout 可以生成多个候选题；正式发布时再处理同源题隔离。

## 3. 可视空间与语义合同

所有 A 和 B/C 所依赖的安全终点复用同一证书：

```text
Habitat body-conditioned navmesh rollout
+ initial-depth view collision rollout
+ swept-corridor coverage from s0
+ oracle_consensus
```

- `collision`：full/depth 标签一致，contact arc 在容差内；R2R 还要求同一非空 initial-visible instance。
- `no_collision`：两套 oracle 都安全，且完整 action corridor 被初始 depth 认证。
- 遮挡、无效 depth、证据缺口、越出可视空间、标签/弧长/instance 不一致均 fail closed。
- `semantic.py` 是 MP3D PLY/house、instance/category、完整 instance triangles 的唯一查询入口。
- `objects.py` 只做初始可见目标筛选与名称/最小编号；`record.py` 只保存 hash-bound atoms。
- B 目标优先使用唯一类别；同类多实例时才在图上加最小编号圆点。

## 4. 最小实现数据流

```text
R2R/MP3D scene
  -> existing random/balanced action collection (main mode)
  -> conseq.v9 records + visible-space certificates + terminal RGB
  -> compile_main_records()
  -> A/B/C eligibility + Closed Exact choices
  -> public/items.jsonl + private/answers.jsonl + private/atoms.jsonl
  -> benchmark.json + report.json
  -> validated static HTML grouped by A/B/C and task ID
```

核心文件职责：

- `pipeline/actions.py`：保留既有随机/均衡 action 生成；无 D action bank。
- `pipeline/collection_runtime.py`：采集 main records 与终点 RGB；无 chain mode。
- `pipeline/semantic.py`：唯一语义与完整目标几何 authority。
- `pipeline/record.py`：A/B/C 所需的 source、certificate、target geometry 与 endpoint atoms。
- `pipeline/benchmark.py`：六个 task 的统一 registry。
- `pipeline/benchmark_tasks.py`：eligibility 和 canonical GT 投影。
- `pipeline/benchmark_builders.py`：只负责题面与选项，不重新计算 GT。
- `pipeline/candidate_preview.py`：编译、验证、exact scoring、报告与 HTML。
- `scripts/build_v16_candidate_preview.py`：可重复传入多个真实 main shard。
- `scripts/eval_benchmark.py`：只评分当前 ABC candidate artifact。

Active collection CLI 只公开 `backend=r2r` 与 `collection_mode=main`。仍被底层 validator 引用的
历史 collection-mode 常量只用于读取旧审计记录，不是当前 benchmark 的采集入口。

## 5. 已删除的冗余

- D chain action、rollout、record、selector、future-pair、CLI 与专用测试。
- 旧 Q1–Q10 formal artifact compiler、writer、validator、evaluator、baseline 与专用测试。
- 旧 near-field/funnel 报告层；底层可视空间安全证书仍保留。
- 旧 human/open/LLM-judge 与 shortcut artifact 脚本。

删除后结构为 44 个 `pipeline/*.py`、约 26.8k 行，低于 50 文件/27.6k 行门。

## 6. 产物与运行

```text
data/candidate_pool/<run>/candidate_qa/benchmark.json
data/candidate_pool/<run>/candidate_qa/public/items.jsonl
data/candidate_pool/<run>/candidate_qa/private/answers.jsonl
data/candidate_pool/<run>/candidate_qa/private/atoms.jsonl
data/candidate_pool/<run>/candidate_qa/report.json
data/candidate_pool/<run>/candidate_qa_report/index.html
```

示例：

```bash
PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python
META0=$(sha256sum data/candidate_pool/<run>/records/r2r/main-r00/run_meta.json | cut -d' ' -f1)
META1=$(sha256sum data/candidate_pool/<run>/records/r2r/main-r01/run_meta.json | cut -d' ' -f1)

$PY scripts/build_v16_candidate_preview.py \
  --records data/candidate_pool/<run>/records/r2r/main-r00/records.jsonl \
  --run-meta-sha256 "$META0" \
  --records data/candidate_pool/<run>/records/r2r/main-r01/records.jsonl \
  --run-meta-sha256 "$META1" \
  --future-view-gate data/candidate_pool/<run>/c1_future_view_gate.json \
  --output data/candidate_pool/<run>/candidate_qa \
  --report data/candidate_pool/<run>/candidate_qa_report

$PY scripts/eval_benchmark.py \
  --benchmark data/candidate_pool/<run>/candidate_qa --gt-as-pred
```

## 7. 验收标准

1. 六个 task 都至少包含真实 R2R case；不足只报告实际 yield，不复制或降门凑数。
2. 所有题通过 visible-space、source、semantic、asset 和 public/private/atom 引用验证。
3. 网页从验证后的 `benchmark.json` 机械生成，每个 case 只出现一次。
4. `--gt-as-pred` 对完整 ABC artifact 的 task/head/overall 都为 `1.0`。
5. 全量 pytest、import-cycle、undefined-name、module/file/line budget 全部通过。
6. D 在重新冻结前不得以任何名称或兼容分支进入 active runtime。
