# EgoConseq-Bench Related-Benchmark Audit

日期：2026-07-30

状态：**SUPERSEDED — historical literature record only; do not cite as the
active EgoConseq protocol.**

取代关系：ABC taxonomy 以
`docs/superpowers/specs/2026-08-03-abc-first-order-consequence-freeze-design.md`
为准；当前实现与 pilot 决策以 `docs/2026-08-07-v12-task-gt-pilot.md` 为准。

作废原因：本文仍使用已删除的 Q1--Q10 taxonomy，并声称四个公开半径、Q4 direct-body
题、reset-matched do(sensor) 及三类已发布干预。这些都不是当前六题 ABC candidate
contract。下文只保留为 2026-07-30 文献核查的历史证据，所有 EgoConseq-facing 对比句、
表格行和 novelty 句均不得直接进入论文。

用途：记录论文页与冻结契约中“与相关 benchmark 的区别”所依据的定向核查。本文档是
文献证据记录，不改变 EgoConseq 的构念、公开字段、采样、评分或发布协议。

## 1. 核查范围与判定字段

本轮优先核查与 EgoConseq 最接近的三条研究邻域：

1. 给定动作预测结果（ActionEQA）；
2. 能力/身体条件导航（CapNav）；
3. 碰撞安全 grounding（TouchSafeBench）。

辅助对照包含当前空间感知、场景记忆、已观察物理过程与广义 world-model
evaluation。每个 benchmark 统一记录：

| 字段 | 判定问题 |
|---|---|
| 核心问题 | 模型到底需要回答什么？ |
| 公开输入 | 模型在预测时实际看到哪些图像、动作、身体或传感器信息？ |
| 动作 | 动作是否给定；是语义动作、机器人控制量，还是导航路线选择？ |
| 输出 | 当前关系、终态、碰撞、路线，还是生成式未来？ |
| 身体条件 | 是否显式公开；是否是题级干预变量；取值是 profile 还是米制几何？ |
| GT 来源 | 人工标注、记录到的真实机器人 episode，还是 simulator 重执行？ |
| 同起点重执行 | 是否在同一初态下只改变 action/body/sensor 并重新执行？ |
| 配对计分 | 是否要求干预对两端联合答对？ |
| 可视支持契约 | 是否在发布前把 physical / observation / evidence GT 分开，并剔除单图不支持的题？ |

“没有相同组合”只在上述字段的**受控合取**上成立；任何单一成分已有先例时都必须明确
承认，不能改写成“首个动作后果 benchmark”或“现有工作都忽略身体”。

## 2. ActionEQA：必须纳入的最近邻

**Primary sources**

- [ActionEQA project page](https://actioneqa.github.io/)
- [TMLR / OpenReview record](https://openreview.net/forum?id=HY2ruqdMt4)

**核查结果**

- 论文：*ActionEQA: Action Interface for Embodied Question Answering*，TMLR 2026。
- 双向任务：
  - State Prediction：给定 starting frame + action，预测 goal state；
  - Action Inference：给定 before/after frames，识别造成状态变化的 action。
- 动作分为 High（task goal）、Mid（semantic motion）与 Low（7-DoF motor command）
  三层。
- 数据来自 BridgeData V2、DROID 和 RT-1 的真实机器人操作 episode。
- 项目页报告 8,795 questions、26,213 unique images、2,629 unique episodes，并评测
  34 个 VLM（26 open-weights、8 proprietary）。
- 项目页报告最佳 Gemini-2.5-Pro overall 58.4%，human 95.6%。这些是
  **ActionEQA 自身结果**，只用于说明其评测规模，不是 EgoConseq 的比较实验数字。

**对 EgoConseq 主张的约束**

ActionEQA 已直接测量 start-frame + action → goal-state prediction，因此：

- 必须称其为 action→state 的最近邻先例；
- EgoConseq 不得主张首创 action-conditioned state/outcome prediction；
- EgoConseq 的差异只能落在更窄的受控协议上。

**仍然成立的协议差异**

ActionEQA 使用既有真实机器人操作轨迹来构造语义—物理动作理解问题。其公开任务不把
米制二维 ground-disc radius 作为题级变量，也不在同一初始状态下重新运行 simulator 来
形成 reset-matched do(action)、do(body) 与 do(sensor) 对；它也没有 EgoConseq 的
Physical GT / Observation GT / Evidence GT 三层分离和 visible-support publication
contract。两者研究问题相邻，但数据机制与可发布主张不同。

## 3. 统一对比记录

| Benchmark / 邻域 | 核心问题与公开输入 | 动作与输出 | 身体条件 | GT 与干预协议 | 与 EgoConseq 的边界 |
|---|---|---|---|---|---|
| **ActionEQA** (TMLR 2026) | 从真实机器人图像理解语义—物理动作接口 | start frame + action → goal state；另有反向 action inference；High/Mid/Low 三层 | 无公开米制 ground-disc radius 轴 | BridgeData V2 / DROID / RT-1 记录 episode；非同起点 simulator re-rollout | 最近邻，证明 action→state 已有先例；差异是四公开半径、同起点重执行、三类配对干预和三层 GT/可视发布契约 |
| **CapNav** ([arXiv:2602.18424](https://arxiv.org/abs/2602.18424)) | 给定 tour video、node graph 和 agent capability，选择可行室内路线 | 路线级 capability-conditioned navigation | 五个离散人/机器人 profile，含尺寸、移动与交互能力 | 逐边 traversability 由人工判断，并用 3D collider 验证；不是 reset-matched rerollout | 身体能力影响路线的先例；不是固定单图、给定度量动作后的终态预测，也没有每题四半径 Direct Body Effect |
| **TouchSafeBench** ([arXiv:2605.31196](https://arxiv.org/abs/2605.31196)) | 从 Habitat 3.0 多视角 RGB-D episode 判断当前安全状态或接触前预警 | 无 EgoConseq 式公开度量移动程序；输出 current/imminent collision class | morphology 是 collision grounding 语境的一部分，但不是公开、逐题改变的四半径轴 | simulator-derived contact labels；主要实验轴是 visual representation × viewpoint，不是 reset-matched body intervention | 碰撞 grounding 的直接先例；不读出同一 locomotion rollout 的终点/目标/未来观测/五探针，也无四半径 Direct Body Effect |
| **Physion / CLEVRER** | 观察完整物理视频后回答接触、解释、预测或反事实问题 | 观察到的过程或场景内物体干预；非 ego metric locomotion program | 无执行者 ground-disc radius | simulator labels；同一仿真底物可派生多读出，但不是公开 action/body/sensor 的同起点重执行 | 为 forced-choice 物理预测、统一底物和反事实读出提供先例；不等同于 EgoConseq 的自身行动协议 |
| **PhysBench** ([arXiv:2501.16411](https://arxiv.org/abs/2501.16411)) | 图像/视频物理理解，覆盖属性、关系、场景与动态域 | 广义物理问答 | 无题级 ground-disc radius | 数据集题目与标签；无 reset-matched locomotion consequence pair | 覆盖面更广；EgoConseq 更窄地隔离 visible-space action consequence |
| **OpenEQA / EgoSchema / VSI-Bench** | 环境记忆、长视频时序理解或视频空间智能 | 回忆/空间问答，而非执行给定动作后的机械终态 | 通常固定或无显式执行者尺寸 | episode/video annotation 或 3D annotation | EgoSchema 应归为长视频时序理解，不应被写成 body-consequence benchmark |
| **BLINK / EmbSpatial / RoboSpatial** | 当前图像中的空间关系、深度、参考系或机器人空间理解 | 静态空间问答 | RoboSpatial 强调参考系，但不提供给定动作重执行 | 标注或几何生成 GT | 为视觉空间能力与参考系纪律提供先例；不测动作执行后的终态 |
| **EgoConseq** | 一张标定 ego RGB + 相机光心高度/FOV + 度量动作 + 可选目标；Q1–Q3/Q6–Q10 给一个半径，Q4 给四个候选半径 | 同一短时 locomotion rollout 的 Interaction、Endpoint & Target Relation、Future Observation、Local Affordance | 身体仅为米制 2D ground-disc radius；相机高度/FOV 是 sensor，不是 body size | simulator 真实重执行；reset-matched do(action/body/sensor)；三层 GT；visible-support publication contract | 贡献是这些装置的受控合取，不是任何单一装置的首创 |

### CapNav 数字的正确解释

CapNav 中 1.00 → 0.22 是 aggregate Human 与 Humanoid profile 的 feasible-task ratio
对比，不是连续半径曲线，也不是同一初态下对半径做 simulator rerollout。页面若引用该
数字，必须同时给出这一限定；更稳妥的正文做法是只写“五个离散 capability profiles”。

### TouchSafeBench 轴的正确解释

TouchSafeBench 明确讨论 viewpoint、robot morphology、metric geometry 与未来碰撞的
绑定，但其主要实验比较视觉表征和视角。不能把它描述成公开 morphology 连续扫描，也不能
据此声称它已实现 EgoConseq 的 Direct Body Effect；反过来，也不能说它“没有身体”。

## 4. Benchmark 与方法论来源分轨

下列工作可以解释设计原则或评价证据强度，但不应混入主 benchmark 对比矩阵：

- world-model evaluation ladder：用于说明 interventional evidence 的层次；
- *Lost in Aggregation*：用于提醒聚合可能掩盖能力差异；
- CheckList、NaturalBench、MMStar、TOMATO 等：分别提供行为测试、配对联合计分、
  盲答/泄漏审计和能力必要性过滤器的设计先例。

它们应出现在“设计依据/审计方法”小节，而不是作为与 EgoConseq 争夺同一任务定义的
benchmark 行。未核实的编号或非 benchmark 方法论文不得被用来扩大 novelty 主张。

## 5. Directed-comparison conclusion

冻结表述与 Design Contract v1.3.3a §3 保持逐字一致：

> Widely used visual-spatial and intuitive-physics benchmarks usually treat the observer's embodiment as fixed or absent; recent body-aware navigation and collision benchmarks do not provide the same reset-matched body-consequence protocol: four public ground-disc radii per direct-body item, transition cases constructed from a privately searched critical radius, and true simulator re-rollouts.

> In our directed comparison of the closest benchmarks, we did not find the same controlled combination: an explicit metric ground-plane disc whose direct effect is evaluated at four public radii, with critical-radius search confined to the private oracle; true simulator re-rollouts; reset-matched do(action)/do(body)/do(sensor); three-layer ground truth separating physical fact, future observation, and visual support; and a preregistered operational visible-support publication contract.

该结论的作用域止于 2026-07-30 已核查的工作及上述字段。正式投稿前应重新执行定向检索，
并把任何更接近的新 benchmark 追加到本文档，而不是静默维持旧 novelty 句。
