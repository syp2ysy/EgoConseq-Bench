# Benchmark Contract

## Scope

EgoConseq 当前测试：给定一张初始第一视角 RGB、body size、camera/FOV 与随机 action sequence，
模型能否预测初始可视空间内的真实后果。

| Head | Task | Output |
|---|---|---|
| A | `A1_collision` | `collision / no_collision` |
| A | `A2_collision_step_grounding` | 1-based action ID |
| A | `A3_contact_object` | first-contact visible category |
| B | `B1_endpoint_distance` | one metric distance choice |
| B | `B2_endpoint_direction` | `front / left / right / rear` |
| C | `C1_future_view_selection` | one future-image choice ID |

## GT

A 使用 body-conditioned Habitat navmesh rollout、initial-depth collision rollout、初始帧
swept-corridor coverage 与 `oracle_consensus`。碰撞还必须由 full/depth 归因到同一个非空、初始
可见 instance。任何隐藏碰撞、不可认证路径或共识漂移均不进入候选集。

B 只使用安全完成的 rollout。目标在 s0 选择；R2R/B1K 使用完整 source geometry，GS 使用
初始帧可见 depth anchor；每道公开 B 题的 `task_metadata.target_geometry_protocol` 明确声明
实际口径，不能把两者混作同一几何真值。B1 的 `b1-metric-choices-rank-balanced.v3` 还冻结全部显示选项及
`plausible_lower_bound_m` / `plausibility_tolerance_m` certificate。C 的正确答案是真实
query endpoint 原生 RGB；candidate 路径确定性生成邻近反事实 action，独立认证后选择三张
互异终点图作为干扰项。终点允许出现初始图中没有的新 observation；四张图保持相同相机高度
与 FOV，B1K 使用一次绑定的 batch transaction 消除渲染历史差异。

记录验证分两层：`validate_record_local` / `validate_file_local` 只检查 JSON、内部证书和
记录输出，不打开 R2R/MP3D source；`validate_record_source_bound` /
`validate_file_source_bound` 再用外部 run-meta/source authority 重放 source facts。正式
collection finalization、candidate compilation、golden 与注册记录检查一律使用 source
层，不能静默降级。Candidate artifact 的 records 与 run metadata 还必须与外部
committed source-authority manifest 精确一致。

C1 candidate 结构冻结为 `c1-counterfactual-selection.v2`，绑定 query、三条反事实程序、
生成器 ID、四张真实终点图与 pairwise block-L1 诊断。候选期不使用像素或语义数值阈值筛题；
产物始终是 pending human review、non-formal、non-headline。正式发布协议仍需独立人工审核
与 held-out confirmation；旧 strict suffix、directed matched/atomic 路径均已删除。

## Scoring

每题 Closed Exact，正确为 1，错误/缺失为 0。正式跨任务汇总是六个 task 的等权宏平均
`six_task_macro`。为兼容候选期诊断，scorer 仍同时输出先在 task 内平均、再在 A/B/C head 内
平均的 `overall`；它不是正式跨任务汇总。任一 task 缺失时，相关 head、`overall` 与
`six_task_macro` 均为 `undefined`，不对幸存项重新归一化。

Candidate artifact 永远 `headline_eligible=false`。`--gt-as-pred` 必须在完整 artifact 上得到所有
task/head、兼容字段 `overall` 和正式字段 `six_task_macro` 均为 1.0。
