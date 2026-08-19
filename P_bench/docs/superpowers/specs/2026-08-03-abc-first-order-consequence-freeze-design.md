# EgoConseq ABC 一阶后果分类冻结稿

日期：2026-08-03

状态：**设计已确认并冻结；R2R candidate pipeline 已实现。**

2026-08-07 implementation note：schema/Task-GT、height-aware family 容量、独立
body near-boundary pilot 与诊断阈值的当前状态见
`docs/2026-08-07-v12-task-gt-pilot.md`。该 amendment 不改变本文的六题 taxonomy；
若本文旧的实现状态或 body-intervention 叙事与 amendment 冲突，以 amendment 为准。

## 1. 文档地位与边界

本文是 EgoConseq 一阶 `A/B/C` taxonomy、正式子问题及 GT 语义的唯一规范来源。
它取代 `2026-08-02-abcd-consequence-taxonomy-design.md` 中有关 A/B/C 的旧定义，尤其取代
`Future Scene Visibility`、单目标可见性与多实体 visibility vector。旧文档仅保留历史参考价值；
后续 ABC 规范不得恢复这些已删除设计。D 当前暂停且未冻结，不得反向改变本稿六个一阶问题。

本文只冻结 benchmark semantics，不声称当前 `egoconseq.qa.v15`、source schema、compiler、
evaluator 或网页已经实现。所有数值门、配额和格式细节必须在后续 QA/implementation contract
中预注册并由 validator fail closed。

## 2. Benchmark 核心与 consequence 判据

EgoConseq 的核心对象固定为：**初始可视空间内的 action consequence**。benchmark 通过显式改变
`action`、`body size` 和 `FOV`，测试 VLM 是否真正理解身体占据、动作几何与可见空间，并据此预测
执行过程、真实终点关系和未来视野。其中 action/body 可以改变 canonical physical consequence；FOV
只改变公开观测、visible-space eligibility 与 future render，不得改写 navmesh collision/arc GT。任何
必须依赖初始公开视野之外的隐藏地图、隐藏障碍或未知实体才能回答的 case 都不进入 candidate benchmark。

公开输入的核心是：一张初始第一视角 RGB、身体尺寸、相机参数和确定性动作序列；目标相关题
还给出一个从初始图像中可以唯一指认的目标。C1 仍只有一张初始 RGB，但会额外提供固定数量的
候选未来 RGB 作为答案选项；这些候选不是额外的初始状态证据。

首版身体合同固定为 ground-disc footprint radius `0.15/0.20/0.25 m` 与全局公开的物理高度
`0.30 m`。逐题仍只需重复其 radius；固定高度写入 public manifest 和全部物理证书，不得与相机
光心高度混淆。radius 或物理高度任一改变，都构成不同的物理身体。

正式问题必须同时满足：

1. **动作因果性**：固定初始状态，改变 action 时答案应当可能改变；
2. **公开输入可判定性**：不得要求模型知道初始 RGB 外的隐藏地图或未知实体；
3. **精确 GT**：答案必须由认证 simulator rollout、几何或终点重渲染机械生成；
4. **反捷径可验证性**：候选与样本分布必须允许 action-only、image-only、位置和频率基线审计。

在静态世界、确定性 ego-motion、首次接触截断的 scope 下，一阶因果结构是：

```text
initial state + action
          |
          v
execution trajectory --------> A · Interaction Consequence
          |
          v
realized clear endpoint ------> B · Endpoint Physical Consequence
          |
          v
terminal camera render -------> C · Future Visual Consequence
```

类别按因果阶段划分，不按答案是“物理”还是“语义”划分。A3 已包含实体语义；C 的本质是
终点传感器观察，而不是任意静态场景语义。

## 3. 冻结后的正式 QA 清单

| 类别 | 正式子问题 | 核心问题 |
|---|---|---|
| A · Interaction Consequence | `A1_collision` | 执行整段动作是否会发生碰撞？ |
|  | `A2_collision_step_grounding` | 已知会碰撞，碰撞发生在第几个 action？ |
|  | `A3_contact_object` | 已知会碰撞，最先接触哪一类可见物体或结构表面？ |
| B · Endpoint Physical Consequence | `B1_endpoint_distance` | 安全完成后，终点到目标的距离是多少米？ |
|  | `B2_endpoint_direction` | 安全完成后，目标相对最终朝向位于哪个方向？ |
| C · Future Visual Consequence | `C1_future_view_selection` | 安全完成后，哪幅候选图是机器人真实看到的终点视野？ |

一阶正式集合固定为六个问题。A、B、C 的子问题数量不需要对称；禁止为了列数对称制造 C2，
也禁止重新引入旧 `Q1/Q2/...` 作为公开 taxonomy 名称。

## 4. A · Interaction Consequence

A 测试动作执行过程中的阻断事件。A 不需要目标；身体尺寸参与 swept-volume/corridor 几何。

### A1_collision

**问题**：

> 以给定身体尺寸执行全部动作，是否会发生碰撞？

题面只问“是否碰撞”，不加入“首次碰撞”等多余措辞。

**答案空间**：`no_collision / collision`。

**GT**：canonical 物理标签来自身体条件化 Habitat navmesh 的完整 swept-corridor rollout；但只有
初始 depth 可见空间对该物理标签提供独立支持时才生成 A candidate。A1、A2、A3 直接复用现有
`view_collision_rollout + corridor_coverage + oracle_consensus`：

- `collision`：navmesh 与 initial-depth rollout 都判定碰撞，首次接触弧长在冻结容差内一致，且从
  起点到首次接触的已执行 corridor 全部通过初始可视空间认证；在 R2R 的稳定实例语义下，full
  contact 与 depth-mask contact 还必须给出同一个非空 instance ID，证明实际碰撞对象存在于初始
  可视空间；
- `no_collision`：两套 rollout 都判定安全，并且完整公开动作 corridor 通过初始可视空间认证；
- coverage 不足、路径出 FOV/被遮挡、碰撞标签不同或接触弧长不同，均删除整个 A candidate，不能
  把 depth 不确定改写成 `no_collision`。

A1、A2、A3 复用同一 visible-space physical certificate 与同一套冻结的起点扰动。A1 只有在
nominal 与全部扰动下的 collision label 和共识门均稳定时才可发布；不得为三个子问题分别挑选
不同扰动来制造稳定样本。

### A2_collision_step_grounding

**条件问题**：

> 已知执行过程中会发生碰撞，碰撞发生在第几个 action？

**答案空间**：公开动作序列中的 `action_1 ... action_N`。选项由公开动作数决定，不由 GT
决定；正式样本至少包含两个具有非零平移 swept segment 的公开 forward action，使碰撞 step
不是单一候选。答案保留它们在原序列中的 action index；连续同向且语义上属于一个动作的内部
simulator 步不得伪装成多个公开 action。

**GT**：首次接触弧长/时间映射回公开 canonical action index。转向若在当前 ground-disc 模型中
不产生新 swept area，则不能被标成碰撞 action。

在继承 A1 shared visible-space certificate 后，与 A1/A3 相同的每次扰动 rollout 都必须碰撞、
映射到同一个原始 1-based action index，且 full/depth 两个接触弧长也必须映射到同一个 action。其
首次接触不能落在相邻 action 的不稳定边界附近。Turn 保留公开 index，但在当前 ground-disc 模型中
不产生平移 swept segment，因此不能成为碰撞答案。稳定性门失败只 withhold A2，不反向否定 A1；
具体边界 margin、证书与数值校验由 implementation plan 冻结。

### A3_contact_object

**条件问题**：

> 已知执行过程中会发生碰撞，机器人最先接触哪一类可见物体或结构表面？

**答案**：由初始可见语义实例支持的具体 contact category，例如 `chair / cabinet / wall`。候选项是
初始帧中所有可接触、可识别类别的确定性去重列表；同类别多个实例只保留一个类别选项，不添加
编号圆点，也不使用 bbox、轮廓线或大面积色块。

**GT**：A3 首先继承 A1/A2 的 shared visible-space physical certificate。身体条件化 Habitat
navmesh 提供 canonical collision 与 first-contact arc；initial depth 对 collision label、contact arc
和已执行 corridor 拥有 fail-closed 的可见证据否决权，但不改写 canonical 物理 GT。MP3D semantic
instance/category authority 只在该共识已经通过后给碰撞实例命名，不作为第二套碰撞物理 oracle。

在冻结的每次 SE(2) 扰动 rollout 中，full geometry 与 depth mask 必须归因到同一个非空实例，且该
实例必须存在于初始 RGB/depth 的可见 inventory；所有扰动实例映射出的 category 必须一致。A3 只
评分这个稳定 category，不要求跨扰动 exact instance 相同，也不再执行 face-level near-tie 或 exact
instance 发布门。category 不一致、不可识别、不可接触、初始不可见或候选类别不足时只 withhold A3；
visible contact witness、depth coverage、full/depth collision label 或 contact arc 共识失败时，
A1/A2/A3 全部 withhold。A2 action-index 稳定性失败只 withhold A2，不污染 A1/A3。

## 5. B · Endpoint Physical Consequence

B 只对 `completed_clear` rollout 发布，测试真实终点与同一初始可见目标之间的物理关系。
目标必须由只读取初始证据的确定性 selector 选择，不能根据终点答案挑目标。

### 目标选择共同规则

- 目标在初始 RGB 中清晰可见且身份稳定；
- 目标 mask、bbox 两边和有效 depth support 足够大，不选择极小、严重截边或深度不稳定目标；
- 初始可见 mask/depth 的 instance key 必须与完整 per-instance triangles authority 指向同一实体；
- 目标位于可回答的中间深度/距离区间，排除极近和极远尾部；
- 场景中类别唯一时直接使用名称；存在多个同类实例时只添加最小化编号圆点；
- selector、编号、名称、reference support 和输入图像共同绑定 hash；
- 所有具体面积、深度、边缘和稳定性阈值由 oracle-only/human pilot 预注册，不得看模型结果后调节。

MP3D 的原始 semantic PLY/house 查询只有一个实现权威：现有 `semantic.py`。它必须提供目标的稳定
instance/category 和完整 per-instance triangles；`objects.py` 只在这些已物化事实上执行可见目标
筛选、命名与编号，其他模块不得增加第二套 PLY decoder。现有由 sampled face centroids 生成、再
确定性截断到最多 256 点的 `target_reference_set` 只可用于 visibility/review，不得作为 B1/B2 或
D2 的正式 GT authority。

### B1_endpoint_distance

**问题**：

> 动作安全完整执行后，机器人终点到标记目标的最近水平距离是多少米？

**GT**：真实终点机器人中心到目标认证 ground-reference support 的最近水平欧氏距离，单位为米。
该 support 必须由 `semantic.py` 返回的目标完整 triangles 与冻结 ground band 的裁剪/投影机械导出；
实现若不做精确最近距离，必须给出逐题误差证书并使误差小于冻结 GT/选项 separation margin。
不得用 sampled/capped reference points 冒充完整或经过误差认证的绝对距离 authority。

目标和动作必须使最终距离落在可测的中间区间，且相对初始距离产生超过 oracle、显示精度和评分
容差联合界的实质变化。正式任务问绝对米制距离，不把 `closer/farther` 设为额外子问题。

### B2_endpoint_direction

**问题**：

> 动作安全完整执行后，目标相对机器人的最终朝向位于哪个方向？

**答案空间**：`front / left / right / rear`。

**GT**：目标 reference centroid 由同一完整 triangles 的三角形面积加权表面质心定义，再投影到
水平面；从真实终点 ego frame 指向该质心计算 bearing，并按冻结扇区边界量化。不得对
sampled/capped points 求均值。目标必须远离扇区边界；四个方向分别具有跨场景支持，不能让目标
几乎总在前方。
B1 的最近 support 与 B2 的 reference centroid 是两个分别冻结的锚点；二者不应被误写成同一个
极坐标向量的严格半径和角度分量。

## 6. C · Future Visual Consequence

C 只对 `completed_clear` rollout 发布，测试 action 导致的整体终点第一视角观察。C 不再使用
单目标 visibility、多实体 visibility vector、图像位置、bbox、面积变化、物体计数或自由场景描述
作为正式子问题。

### C1_future_view_selection

**问题**：

> 动作安全完整执行后，哪一幅候选图像是机器人最终真正看到的视野？

**答案空间**：固定数量候选未来视野中的一个 image ID。候选数量与正式随机 null 在 QA contract
中统一冻结，不能按题目难度或 GT 改变。

**GT**：在认证 `completed_clear` 真实终点，以相同 backend、相机高度、HFOV/VFOV、分辨率和渲染
设置生成的 terminal RGB。错误候选也必须来自同一 scene、通过同一证据门的认证
`completed_clear` counterfactual rollout，并与主 action 具有相同公开 action 数、相同 planned-path
length bucket 和可比的 net-yaw/displacement bucket；例如等幅反向转弯、顺序置换或其他匹配的
action sibling。不得使用 collision-truncated、不同相机/渲染条件、随机场景、直接复制初始帧或
明显不同动作预算的终点作为主候选。

**可想象性与反捷径门**：

- 正确终点视野必须有足够内容可由初始公开视野证据支持；私有 depth/reprojection 只用于认证，
  不作为模型输入；严重依赖全新房间、全新物体或大面积未知区域的样本不发布；
- 候选应共享场景身份和粗粒度外观，并匹配 planned length、net yaw、ego displacement 与初始视野
  overlap 的冻结分桶；C1 不依赖一个未公开的 B-target；
- 正确位置、counterfactual family、动作长度和视觉差异程度必须平衡；
- action-only、initial-image-only、candidate-image-only、位置和最近帧相似度基线必须在发布前审计；
- 同一初始状态的 action-intervention siblings 必须保持同一 split，并证明改变 action 会改变正确图像。

候选图必须通过冻结、可机械重算的视觉差异门：既不能近似重复，也不能用无关远景制造显然错误的
干扰项；该门直接运行在认证的原生 RGB 上，不允许 resize。具体度量、整数证书、calibration-only
流程和 gate asset schema 由 implementation plan 规定，不属于 task taxonomy。

Candidate 阶段可以生成 C1，并私有标记 human gate 尚未完成；正式发布前，C1 必须独立通过在读取
human response 前预注册的 task-level human-answerability gate。不得根据逐题人类正确性删除题目后
重算通过率。D3 使用独立 gate，不能与 C1 pooled 通过；完整 selector、统计和签名协议属于 formal
release 计划。

C1 是已有 future-observation/world-model 任务形式在 EgoConseq 因果管线中的使用，不声称题型本身
新颖。贡献来自它与同一次认证 rollout 的 A/B、body/action/camera 条件和反事实控制共同组成统一
benchmark。

## 7. 条件路由与正式输出

一个 source rollout 可以产生多个候选 QA。下表中的 `collision` 与 `completed_clear` 均指已经通过
shared visible-space certificate 的结局；盲区碰撞和驶入不可认证区域的安全路径都不进入 candidate
pool。Candidate pipeline 保留全部认证 siblings 以支持
funnel、可视化和后续发布选择，并始终 `headline_eligible=false`。

| rollout 结局 | 可生成的正式问题 |
|---|---|
| `collision` | A1、A2、A3 |
| `completed_clear` | A1、B1、B2、C1 |

不得把 private execution regime 写入公开题面。A2/A3 的条件由题面明确给出；B/C 的正式来源只取
安全完成样本。ABC 不附加派生诊断问题，candidate QA 目录只输出上述六种正式 task ID。

题型的公开条件固定为：A2/A3 明示 full program 会碰撞；B1/B2/C1 来自 full program 安全完成。
由此产生的跨题存在性泄漏必须在 formal release
阶段按相同 scene、s0、完整 body 与 canonical action-prefix 关系统一隔离：安全长程序蕴含其前缀安全，
碰撞前缀蕴含其扩展程序会碰撞。Candidate pipeline 不删除这些题，正式发布计划再冻结 conflict
graph、selector 与 split 规则。

同一个 clear rollout 的 B1、B2、C1 可以共同生成；它们共享 endpoint 所产生的答案相关性不属于
仅扫描题库即可推出答案的 premise leakage。

## 8. 完备性声明与明确排除项

在“静态世界、确定性短时 ego-motion、首次接触截断、单张初始 RGB 公开证据（C1 另给固定候选
未来图作为答案选项）”的合同内，A/B/C 分别覆盖过程事件、终点物理关系和终点视觉观察，因此
一阶 taxonomy 在该 scope 内完备。

它不声称覆盖全部 embodied consequence。以下内容被有意排除：

- 接触后物体的位移、倾倒、开闭、破损或其他环境状态变化；
- 速度、摩擦、制动、反弹、滑动、损伤等动力学；
- 人物或其他智能体的反应；
- 音频、触觉及其他传感模态；
- 隐藏地图中的新物体、完整自由文本未来场景和不可唯一判定内容；
- 给定 continuation 的二阶 chained consequence、闭环观察、后续动作选择、best-action selection
  和 planning 当前均被排除，D 待重新设计。

若未来引入带可靠动力学参数和对象状态的 simulator，环境/物体状态变化应新增独立后果类别，
而不是伪装成当前 C2。
