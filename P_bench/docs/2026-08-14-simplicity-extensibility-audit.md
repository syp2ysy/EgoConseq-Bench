# 简洁性与可扩展性审计（2026-08-14 收口版）

本文记录 `P_bench` 在清除 pose-pool、family、body-intervention、HM3D
等退役路径后的增量审计及本轮处置。目标是消除真实重复，不以代码行数为目标
抽象语义不同的实现，也不增加超出 `AGENTS.md` 本地离线威胁模型的防御。

## 已完成

- 退役功能及其生产分支已删除；pipeline 约从 44k 行降至 35k 行。
- collection finalization seal 后不再重复做整份 source-bound record 验证。
- candidate compilation 复用一次解码结果，不再反复扫描相同 records。
- `collection_runtime` 已改用共享的 `prune_backend_scoped_params`，child 与
  supervisor 不再维护两份 backend 参数裁剪规则。
- collection source-key 已收进 `DatasetSourceContract`；原有六份生产映射归一。
- collection dataset 顺序由 `main_collection_datasets()` 唯一提供，并固定为
  `r2r, gs, b1k`。capability report 的展示顺序独立保留。
- B1K completed/partial terminal validation 共用一个本地实现，两个语义清晰的
  薄包装继续保留。
- 12 个零引用的 GS/family/capacity/reprojection 常量已删除。它们只会改变新
  collection 的 config digest；sealed records 与既有 QA 不会用当前 config
  重算旧摘要。
- 唯一未 sealed 且未进入 9K QA source map 的 R2R shard 已放弃，没有新增旧
  config resume adapter。
- active Golden 已换成 R2R/GS/B1K 三库、六任务覆盖；旧 manifest 仅作历史
  provenance。

## 审计结论

### 保留的科学验证不是冗余安全代码

以下内容直接定义 benchmark GT，必须保留：

- full/depth rollout consensus；
- swept-corridor visible evidence；
- 七扰动稳定性证书；
- source asset 与 record 的单次 SHA 绑定；
- 一个 source-bound 三数据集 Golden 和 `--gt-as-pred` scorer 检查。

它们防止 GT 漂移或 source 绑错，属于仓库明确威胁模型，不是 hostile-filesystem
防御。

### 明确不合并的小模块

- 三个 bootstrap 实现计算的统计量不同：pooled mean、multi-predictor accuracy、
  familywise maximum。为节省几十行引入 callback framework 会更复杂。
- 四个 audit CLI 各自入口清晰、参数不同；合成一个子命令不会降低核心复杂度。
- `collection_mode` 虽然目前只有 `main`，仍是 record/run contract 的明确不变量；
  删除它的迁移成本高于约二十处检查的维护成本。

### Backend 分支数量只是诊断指标

共享模块中仍有显式 R2R/GS/B1K 分支，其中大量分支编码真实 authority 差异，
不能机械替换为 `getattr`。尤其 B1K simultaneous C1 batch 等 mandatory capability
必须显式 fail-closed。

在没有第四个后端需求时，不引入 BackendRegistry 或 Session Protocol。第四个
后端明确后，再以其真实能力为输入设计注册表；届时目标是新增注册项和独立
adapter，而不是提前猜测通用接口。

## 当前结构性约束

- `DatasetSourceContract.collection_source_param` 是 controller source 路由的唯一
  数据源。
- `main_collection_datasets()` 的顺序是调度契约：`r2r, gs, b1k`。
- `capability_contracts.DATASET_ORDER` 仅控制报告展示，允许与调度顺序不同。
- active Golden 只使用 41 条 records，而不是完整 9,369 QA；R2R 小 shard 预期
  不贡献 A1，GS/B1K 提供 A1 回归覆盖。

## 后续原则

1. 发现重复时先证明两份代码表达同一规则，再合并；不要按相似外观抽象。
2. 已 sealed artifact 在同一进程链中只验证一次，后续只比文件摘要。
3. 新增安全防御前必须先指出它覆盖 `AGENTS.md` 中哪一条威胁。
4. 没有第四个数据集需求前，不进行大规模 backend 分发重构。
5. 日常交付使用 targeted tests，收口时执行一次完整 pytest 和一个 active
   Golden，不做无意义的多轮 smoke。
