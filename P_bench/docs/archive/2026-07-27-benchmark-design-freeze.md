> **Superseded — archived for provenance, not current.**
>
> This is a Q1--Q10 era freeze contract. The active taxonomy is the six
> ABC tasks A1/A2/A3/B1/B2/C1; the normative upstream is
> `docs/superpowers/specs/2026-08-03-abc-first-order-consequence-freeze-design.md`
> and the implementation plan is
> `docs/superpowers/plans/2026-08-04-r2r-abc-candidate-pipeline.md`.
> Entry points named below (serve_benchmark.py, benchmark_viewer.html,
> serve_viz.html, the Q1--Q10 artifact stack) no longer exist. Do not
> implement anything from this file.

# EgoConseq-Bench Benchmark Design Contract v1.3.3a — design-final

（设计契约；pilot 数值冻结后升 v2.0 正式 Publication Contract）

日期：2026-07-27（v1.3.2 / v1.3.3 修订：2026-07-28；v1.3.3a 修订：
2026-07-30）。状态：**v1.3.3a —— 设计已裁决
（design-final）**。八项开放设计问题已在 §13 正式裁决，构念与**协议分支**均已冻结。
定向采集后的漏斗/option-only 审计只做两件事：确定数值（网格、灰区、配额、Tier 判定），
并依**预注册规则**在既定分支中选择——Q3 exact-category 是否升格为 secondary scored
variant（superclass primary 固定，§6.8）、Q6 geodesic diagnostic → scored。
**不产生新协议。**

**v1.1 修订记录（相对 v1）**：
1. **范围收窄（用户决定）**：benchmark 只讨论初始可见空间。凡答案依赖视野外/被遮挡空间
   的问题**一律不发布**；不再测弃权。`insufficient_evidence` 从公开选项中全部移除，
   Evidence Responsiveness headline 与 R3 主张删除（→ §5.4 修订 4）。
2. 修复六个定义问题：claim matrix（§8.1）、null-corrected 聚合（§8.2）、Q5a 聚合与缺失
   策略（§8.4）、审计规则可执行化（§9）、选项协议表述（§6.8）、propagation 第一门措辞（§7.2）。
3. 修复措辞与事实错误：motivation 过强句、像素级 WM 过强句、Q5 scaffold 描述、
   50/50 表述、文末多余围栏。
4. 评审修正五处："certified answerable" 降级为操作性契约措辞（§0/§1/§4.3）；
   action-conditioning 检验改为正式 do(action) 配对、废除 "matched wrong action"
   概念（§9.2）；选择偏置审计改为"筛选前后增益差"触发、能力信号不触发再平衡（§9.3）；
   null 校正定为 item 级 `null_i`（§8.2）；Q3 选项等价断言降级为待审计命题（§6.8）。
   另：Q5a required heads 明确化（§8.4）、Integration 不参与 headline 措辞（§5.4）、
   human gate 按 head × variant（§8.6）、ablation 不宣称满足完整契约（§5.2）。
5. **v1.2**：§13 八项设计裁决（身体范围=平面 footprint、弃权取消确认、Propagation GT
   =半径无关的中心到固定目标参考集距离、Q4 数值程序、Tier 规则、sensor 配对门、
   Q6 geodesic=WR variant 有条件计分、headline/body 叙事）；定位语收窄为
   static 3D environments；首次漏斗审计数字入档（§13.0）。
6. **v1.3 —— 焦点重构（单一底物）**：显式定义唯一核心问题与 target-conditioned
   rollout group 底物 + canonical future-state vector，所有题目族改述为该向量的投影
   （§2 焦点结构；设计逻辑先例以 **MLLM/VLM 时代**为主：MMStar / NaturalBench /
   HallusionBench / TOMATO / PhysBench / EmbodiedBench / RoboSpatial + 既有
   VSI-Bench / BLINK / BEAR，均本轮定向核实；CLEVRER 底物、Physion 同刺激、
   IntPhys/Winoground 最小对单列为经典根源）；新增**逐 head 必要性过滤器表**
   （受 TOMATO 必要性原则启发，含 Local Affordance 的起点/终点掩码过滤器）；采纳 MMStar
   的 multi-modal gain / leakage 报告名；World Relation 改名 **Endpoint & Target
   Relation** 并纳入终点位姿诊断读出；距离语义拆三量（target_range /
   target_clearance / target_camera_depth），Q6 主读出切换为 target_range（与 §13.3
   传播 GT 同定义）；目标绑定成为 validator 强制项；可视契约新增目标视线行；
   Q8 增加消失原因分层与两类设计配对。
7. **v1.3.1 —— 定义收口（评审五处 + 文献三处）**：底物拆两层（base_rollout_key 与
   target_projection_key，Q1/Q2/Q3/Q9 不再按目标重复）；目标参考集五条实现级定义
   （私有完整几何 / 稳定 instance / 跨 sibling 恒等 / 世界坐标 / validator 哈希校验），
   审计质心降为 proxy；Q7 必要性过滤器升为答案级旋转反事实
   （sector(actual) ≠ sector(yaw-only)），Q6 指出旋转反事实对 range 空洞、改用方向
   启发式反事实的两档结构；Q9 增加 matched-action 掩码对为正式 action-conditioning
   证据；Q8 两类配对按干预类型分立（FOV 对 = do(sensor) 型，遮挡对 = matched
   do(action) 型）且一律取 realized endpoint；PhysBench 措辞改正（含动态域）；
   TOMATO 改为"原则启发"；within/cross-scene 改称 generalization gap。
8. **v1.3.1a —— 文档收口（四处）**：do(sensor) 拆 height-only / FOV-only 两因子
   （高度改视点与遮挡，FOV 只改视锥，分别报告）；information ablation 冻结四条件
   （full / no-height / no-FOV / no-both，height 效用独立可归因）；Q6 子集改名
   **direction-heuristic-resistant** 并收窄解释（点积符号失败 ≠ 完整 3D 关系推理必要）；
   状态措辞改为"数值 + 预注册分支"；文档身份统一为 Design Contract，v2.0 才是正式
   Publication Contract。
9. **v1.3.2 —— 诊断增强 + Q3 保守默认（窄幅协议修订；构念、四头分类、干预体系与
   primary evaluation 全部不动）**：
   (1) 新增 §8.8 oracle 分解梯子与公共几何管线基线：Public Geometric Pipeline（公平
   baseline）/ Oracle-Depth / Oracle-Endpoint（Interaction 整头不适用）/ Full-Geometry
   Oracle（sanity ceiling），解释为 paired diagnostic gains、禁机械相加；
   (2) §9.2 corrupted-prompt 从"可选"升为常设 **Input-Binding Diagnostics**
   （Action/Body Prompt-Binding Corruption），定性三不变：不进 Action Fidelity、
   不作 construct-validity 门、不作 release gate；
   (3) Q3 closed primary 固定为**接触 superclass 词表**；exact category 默认 open
   diagnostic，最多经预注册七门**单向升格**为 additional scored secondary variant，
   永不替换 superclass primary（§6.1/§6.8/§12.1/§12.2 同步）；
   (4) claim–evidence 措辞收紧：§3 残留 "certified" 统一为 preregistered operational；
   增补 WM-ABench / World Models in Words 对比行与四条"不能主张/可以主张"红线；
   §0 增补证据边界冻结句。
10. **v1.3.3 —— 执行状态路由（preview 退化审计触发的协议修订）**。依据
    （2026-07-28 preview full 条件与 Q1 真值的配对审计）：Q9 forward_safe 174/175、
    turn_choice 173/173 落在碰撞 rollout 上且标签坍缩为 unsafe / neither（退化为
    Q1 换皮，majority ≈ 99%）；Q6 closed 153/153 全 closer；Q1 池 183 collision /
    57 safe；根因 = `_realized_state_eligibility` 放行碰撞 rollout 进 endpoint 头
    （benchmark.py:686），且 Q9 掩码过滤器（:1013）在无碰撞前提下系统性选中碰撞终点。
    修订六项：
    (1) 新增非评分属性 **`execution_regime ∈ {completed_clear, contact_truncated,
    invalid_geometry}`**（§5.5），validator 从私有 rollout 独立重算；
    (2) **执行状态路由**：Q6/Q7/Q8/Q9 primary（core）只发布 `completed_clear`
    rollout；`contact_truncated` 的下游读出移入 Post-contact Consequence
    Propagation 诊断 track（§6.10），标 `post_contact_diagnostic`，不进四头
    headline；Q1/Q2/Q4 两种 regime 均可、Q3 仅已归因碰撞（原规则不变）；
    **禁止把 Q6/Q8 的选项动态改成碰撞类别**（§6.8 选项-GT 条件独立原则的直接推论）；
    (3) **Q6 closed 收为 closer / farther 二分类**：GT 严格按 target_range 差值符号；
    unchanged 在纯符号 GT 下仅精确零可达，属死选项，移出公开选项集；精确零样本降为
    限量 diagnostic control；灰区照旧只决定发布、不改写标签；
    (4) **Q9 六条资格门**（§6.4），主动作无碰撞完整执行为前提；
    (5) **发布硬门七条**（§12.4）；当前 preview 定级 **candidate**，其 core 标记无效；
    (6) §13.9 定向采集补五项（安全 toward/lateral/away、安全 Q9 掩码变化、双安全
    matched 掩码对、Q8 安全配对、Q1 标签配额）。
11. **v1.3.3a —— 文献与措辞窄幅修订（2026-07-30；不产生新协议）**：
    将 ActionEQA（TMLR 2026）补入定向对比并明确其是
    start-frame + action → goal-state 的最近邻先例；把公开 Q4 描述收紧为
    **每题四个公开 ground-disc 半径**，临界半径连续搜索仅属于 private oracle 构题过程；
    同步修正 CapNav / TouchSafeBench 的比较口径。构念、公开字段、采样、评分、
    发布门与实现契约均不改变。

**与 archive/2026-07-25-sampling-freeze.md 的优先级**：本文档 supersede 该文档的
reporting axes、P0-9 capability map、P0-9 双 headline 条款、Q10 聚合归属；
其余条款（P0-1/P0-2 采样纪律、层次聚合原则、margin/灰区纪律、matched pairs、
missing-family 禁静默重归一化）继续生效。

---

## 0. 一句话主张（论文用，英文冻结）

> EgoConseq-Bench evaluates whether a VLM can act as a visually grounded, body-conditioned
> consequence model: given one calibrated egocentric RGB image, an explicit body footprint,
> and a metric action program, predict physically grounded short-horizon consequences under
> controlled interventions on action, body, sensor, and calibration information. Every
> published question satisfies a preregistered operational visible-support contract;
> questions whose required geometry falls outside that contract are excluded.

克制条款：本 benchmark 评估的是**行为层面**的 action-conditioned consequence 能力，
不主张证明模型内部维护了完整 world model。

**证据边界（v1.3.2 冻结，英文）**：

> Our evidence concerns behavioral outcome fidelity and counterfactual responsiveness.
> It does not establish that a model internally represents a complete world model or
> possesses long-horizon planning and policy-evaluation capabilities.

**定位语（冻结）**：A benchmark of **visible-space, body-conditioned action-consequence
prediction in static 3D environments**. 不自称一般"物理世界模型 benchmark"。

**范围声明（v1.1 核心）**：我们只讨论可视空间。超出当前视野的空间不出题、不要求模型
推理、也不测"模型是否知道证据不足"。证据充分性（Evidence GT）保留为**私有发布门**，
不是被测能力。

---

## 1. Motivation（三层递进）

**应用层。** VLM 正被接入具身系统充当规划"大脑"，但现有评测很少直接回答前置问题：
它能否从第一视角预测"**我这个身体**、做**这个动作**、会发生什么"。最近的 body-aware
评测触及了部分要素——TouchSafeBench（2605.31196）在 morphology-aware 碰撞语境中
主要比较 visual representation × viewpoint 下的当前/即将碰撞分类，CapNav
（2602.18424）测能力条件化的路线导航——但均未提供固定单图下
每个 Direct Body Effect item 公开四个 ground-disc 半径、由 private oracle 搜索临界半径
构题、并在同起点进行 simulator 重执行的后果协议。

**科学层。** 物理后果有两个被系统性忽略的性质：

1. **body-conditioned**：同一场景、同一动作，半径 0.2 m 的身体过得去、0.3 m 的过不去。
   这是具身智能区别于旁观者视觉的定义性性质，但常用评测把身体当常量或没有身体。
2. **evidence-bounded**：物理上有确定答案，但单张图未必支持这个答案（拐角看不见）。
   v1.1 对这个性质的处理是**发布契约**而非被测能力：发布集由预注册的操作性
   visible-support contract 筛选，契约不满足的不发布（§4）。"人类可答"不由几何契约
   担保，由 human gate 按 head × variant 另行验证（§8.6）。

**方法层。** 我们的读出在**后果级**：GT 是可判定的二/三值物理事实，不是感知相似度。
因此可以做真正的 do() 干预实验。许多像素级短时预测评测在"动作可控性"层仍主要依赖
感知代理指标（Δ-LPIPS、round-trip consistency 等；WM 评估阶梯 2606.15032 称该层为
"interventional threshold"）；我们在同一层给出可审计物理真值上的精确干预。

---

## 2. 构念与范围

**构念**：在满足预注册可视支持契约的空间内，VLM 能否将 3D 场景结构、
标定信息、自身身体尺寸和度量动作程序结合，预测动作执行中的物理交互、实际终点、
未来观测和局部可行性。

三个不可拆开的绑定：

1. scene geometry × body geometry（能否通过/碰撞）；
2. scene × body × metric action（停在哪、撞到什么、终点状态）；
3. physical truth × visual support（只有可见空间支持的物理事实才进入发布集）。

### 焦点结构（v1.3 核心）：一个问题、一个底物、一条推导规则

**唯一核心问题**：

> 这次执行之后，世界相对我会是什么样？
> （单图 + 平面身体 + 度量动作 → 未来 ego 状态 + 目标相对状态 + 未来观测 + 局部可行性）

**单一底物，两层键（v1.3.1）**：

```
base_rollout_key       = (scene, pose, sensor profile, body radius, action)
target_projection_key  = base_rollout_key + target_instance_id
```

base future state（每 base_rollout_key 一份）：

```
collision, executed_action_prefix, stop_stage,
realized_endpoint {translation_m, heading_change_deg},
terminal_physical_safe_mask[5]
```

per-target projection（每 target_projection_key 一份）：

```
target_range_before_m, target_range_after_m,
target_bearing_after_deg, target_camera_depth_m,
target_visible_after {visible, center_norm, area_ratio, invisible_cause}
```

| 来源 | 任务 |
|---|---|
| base rollout | Q1 / Q2 / Q3 / Q9（**不因目标重复生成**） |
| target projection | Q6 / Q7 / Q8 |
| base + 一个固定目标 | joint-vector 读出、Body Propagation |
| 多候选动作 + 固定目标 | Q10 |

**推导规则（三条，全部机械）**：
1. 每个题目族 = 上述两层状态的一个**投影**（Q1=collision；Q2=stop_stage+prefix；
   Q6=target_range 变化；Q7=target_bearing_after；Q8=target_visible_after；
   Q9=safe_mask；joint-vector 读出 = 整个向量）；
2. 每个干预 = 对底物**输入键**的一个 do()（action / body / sensor / 信息可见性）；
3. 每个发布门 = 可视支持契约在该投影**依赖集**上的实例化（§4）。

没有任何题目族拥有独立于底物的语义——这是对"松散 VQA 题库"的结构性否定。

**设计逻辑先例（v1.3 定向调研，均已核实；以 MLLM/VLM 时代为主）**：

| 先例（评测对象 = 现代 VLM） | 焦点问题 | 核心装置 | 我们的对应 |
|---|---|---|---|
| MMStar (2403.20330) | 样本不看图也能答 + 训练泄漏 | 逐样本 vision-criticality 筛查；**multi-modal gain / leakage 双指标** | −image ablation 报 per-head multi-modal gain；within/cross-scene **generalization gap**（仅有额外污染证据时才称 leakage） |
| NaturalBench (2410.14669) | 盲答先验击穿 VQA | **一题配两图、答案相反**的 vision-centric 配对；两端全对才计分 | reset-matched do() 配对 + pair-exact 计分（同构） |
| HallusionBench (2310.14566) | 语言幻觉 vs 视觉错觉纠缠 | 专家构造**控制组**（原图/改图问答对），question-pair accuracy | sensor siblings（Q5a 配对）+ 信息消融控制组 |
| TOMATO (2410.23266) | 时序能力被高估 | **必要性三原则**（multi-frame gain / order sensitivity / information disparity）——每题须过可测的能力必要性过滤器 | 任务特定必要性过滤器（原则启发，见下表） |
| VSI-Bench (2412.14171) | 视频空间智能 | 3D 标注驱动生成 + 盲答人工验证 + Frequency baseline | human gate + majority/UCB 门 |
| BLINK (2404.12390) | 核心视觉感知 | 收题准则 = "人一眨眼能解 + 语言不可中介" | 收题准则 = 可视契约可判 + human gate 可答 |
| BEAR (2510.08759) | 原子具身技能 | 能力隔离子集防总分掩盖 + 组合类单列 | 四头 macro + 切片，Integration 单列 |
| PhysBench (2501.16411) | VLM 物理理解缺失 | 属性/关系/场景/**动态**四域 19 子类覆盖 | 差异在协议而非领域：显式身体 + metric action 的 reset-matched 自身后果干预 |
| EmbodiedBench (2502.09560) | MLLM 具身 agent | 高/低层分级 + 能力隔离子集 | 同构（head 隔离 + rollout_stage 属性） |

经典装置根源（一行说明）：CLEVRER 的"同一仿真底物出多种读出"、Physion 的"统一
forced choice + 人机同刺激"、IntPhys/Winoground 的最小对——分别是本设计底物、读出
与配对机制的历史出处。

**必要性过滤器（受 TOMATO 能力必要性原则**启发**的任务特定过滤器——非其三指标的数学实例；v1.3.1 冻结）**——每个 head 必须声明
一个可测的过滤器，证明被测能力是答对的**必要条件**：

| Head | 被测能力 | 必要性过滤器 | 状态 |
|---|---|---|---|
| Interaction | 动作条件化 | matched action pair：同图同复杂度、仅扫掠走廊不同、标签不同 | 已有（P0-2） |
| Endpoint & Target Relation | 平移视差 / 非轴向推理 | **Q7（答案级）**：sector(GT_actual) ≠ sector(GT_rotation-only)，数值 parallax 保留为 margin；**Q6**：旋转反事实对 range 恒为 unchanged（空洞），改用方向启发式反事实——GT ≠ sign(位移·目标方向) 的 **direction-heuristic-resistant** 子集单独计分（只证明抗方向启发式，不宣称完整 3D 关系推理必要：d_f²−d_i² = ‖Δ‖²−2Δ·t，点积符号失败不蕴含须恢复完整场景），其余保留抑制门 + action-only 攻击实测失败 | v1.3.1 升级 |
| Future Observation | 视野边界 vs 遮挡理解分离 | **FOV 对**（do(sensor) 型：固定世界/pose/action/body/target/物理终点，只变 FOV）与**遮挡对**（matched do(action) 型：固定图/target/body/sensor + 复杂度 + 终点 bearing 桶，变走廊/视线） | v1.3.1 按干预类型分立（§6.3） |
| Local Affordance | 终点推理 + 动作条件化 | ① endpoint 掩码 ≠ 起点掩码（单题必要性/yield 过滤器）；② matched 动作对 A/B 产生不同 endpoint 掩码 + pair-exact（**正式 action-conditioning 证据**） | v1.3.1 两级化 |
| do(body) | 身体条件化 | 四半径真翻转（Direct）/ Δendpoint 过灰区（Propagation） | 已有 |

**WM 阶梯定位**（2606.15032 的 L0–L7）：

| 我们的成分 | 阶梯位置 |
|---|---|
| 绝对后果预测（无干预） | L3 physical/outcome fidelity |
| reset-matched do(action) | L4 counterfactual action fidelity |
| do(body) | 与 L4 同类，但该阶梯**未单列**的 embodiment 干预轴 |
| 不主张 | L5–L7（策略评估/排序/优化） |

**范围裁剪（论文须主动声明）**：

- 读出为 descriptive outcome / predictive consequence / counterfactual response；
  **不提供 CLEVRER 式 explanatory readout**（Q3 是接触归因，不是因果解释）；
- **不评估不确定性表达与弃权校准**（v1.1 移出范围；future work）；selective
  prediction / safety-calibration 叙事不进入论文主张；
- 场景为**静态**：不测动力学、力、可变形物体、移动物体或复杂物体交互；
- "body-conditioned" 的准确含义是**地面平面 footprint（圆盘半径）条件化**，不主张
  三维身体/高度/形状理解（body height / 3D cylinder oracle = vNext，§13.1）；
- 不评估生成式未来视频、长时闭环规划、策略优化、不可见空间探索；
- 机器人是 **2D ground disc，半径是唯一身体变量**（`docs/archive/2026-06-goal-origin.md`，origin 文档）；
  `body_height_m` / 3D cylinder oracle 属 vNext，在 oracle 支持竖直碰撞之前
  不得公开身体高度字段。

---

## 3. 与现有工作的对比

| 工作 | 场景/构念 | 关键设计装置 | 与我们的差异 |
|---|---|---|---|
| VSI-Bench (2412.14171) | 视频空间智能 | 按被测量性质分 3 类 8 任务；盲答人工验证 + 双向追溯；Frequency baseline（12 开源模型 7 个低于它） | 无身体、无动作 rollout、无后果 |
| BEAR (2510.08759) | 具身原子技能 | 6 类 14 技能；类内难度校准；Long-horizon 只做组合 | 无度量身体、无反事实干预、GT 多为标注非仿真 |
| PhysBench (2501.16411) | VLM 物理世界理解 | 属性/关系/场景/动态四域 19 子类，10k 条 image-video-text | 覆盖广义物理理解（含动态）；无显式自身身体、无 reset-matched 后果干预 |
| RoboSpatial (2411.16537) | 机器人空间理解 | 3D 扫描 + ego/world/object 参考系标注，1M 图 3M 关系 | 空间**关系**问答，无度量动作执行与后果；借鉴其参考系纪律——我们显式声明读出参考系（bearing = 终点 ego 系，range = 世界米制） |
| Physion (2106.08261) | 第三人称物理直觉 | 8 类场景**统一二值读出**（两物体是否接触）；人机**同刺激**；**forced choice 无弃权** | 旁观者视角、无自我身体、无动作程序。为 forced-choice 物理预测与人机同刺激比较提供先例（不为我们的筛选契约背书） |
| CLEVRER (1910.01442) | 视频因果 | 四读出分离（descriptive/explanatory/predictive/counterfactual） | 第三人称；其 counterfactual 移走物体，我们的改变自己 |
| ActionEQA (TMLR 2026) | 机器人动作理解 | 双向任务：start frame + action → goal state（State Prediction），以及 before/after → action（Action Inference）；High/Mid/Low 三层动作 | **action→state 最近邻先例**，所以我们不主张首创动作结果预测。其样本来自 BridgeData V2 / DROID / RT-1 真实机器人操作数据；无显式米制 ground-disc radius、无同起点 simulator re-rollout、无 reset-matched do(action/body/sensor)、无三层 GT 与可视支持发布契约 |
| TouchSafeBench (2605.31196) | 碰撞安全 | 绑定 viewpoint、robot morphology 与 metric geometry；主要实验改变视觉表征与视角 | 多视角 RGB-D episode；当前/即将碰撞分类。morphology 是碰撞 grounding 的上下文，不是公开的四半径 Direct Body Effect 轴；也不提供给定度量移动程序的同起点重执行 |
| CapNav (2602.18424) | 能力条件导航 | tour video + node graph；五个离散人/机器人 capability profiles；逐边 traversability 由人工判断并用 3D collider 验证 | 路线级 capability-conditioned navigation；不是固定单图的同起点 simulator re-rollout，也不测试每题四个公开半径的 Direct Body Effect |
| WM-ABench (2506.21876) | VLM 内部世界模型原子评测 | 23 原子维度、6 模拟器、受控反事实 | 覆盖广义 WM 维度；无"单图 + 四个公开 ground-disc 半径 + metric action 真实重 rollout"组合。**禁我方自称首个系统 WM 评测** |
| World Models in Words (2605.29585) | 物理状态转移 trace 审计 | 单图 + typed trace（初态/转移/终态）混合验证器 | 审计语言 trace 的内部一致性；我们的 terminal ego-state 由真实 simulator 重 rollout 机械导出，非标注/语言 trace。**禁我方自称首个结构化状态转移评测** |
| PHYRE (1908.05656) | 2D 物理解谜 | 以跨谜题泛化为评估目标 | split 纪律先例（§8.6 within/cross-scene gap） |
| WM 阶梯 (2606.15032) | 评估方法论 | L0–L7 证据阶梯 | 定位来源（§2） |
| Lost in Aggregation (2606.22219) | 空间认知分层 | 实测层级不单调 | **支持性证据**（不同设定的预印本），是 rollout_stage 不评分的佐证之一，非决定性依据 |

（SQuAD 2.0 / VizWiz 的弃权先例随 R3 移出范围不再进对比表；其"发布筛选会引入表面
相关性"的教训转化为 §9.3 的选择偏置审计。）

**冻结的差异声明措辞**（英文，不用"first"）：

> Widely used visual-spatial and intuitive-physics benchmarks usually treat the observer's embodiment as fixed or absent; recent body-aware navigation and collision benchmarks do not provide the same reset-matched body-consequence protocol: four public ground-disc radii per direct-body item, transition cases constructed from a privately searched critical radius, and true simulator re-rollouts.

> In our directed comparison of the closest benchmarks, we did not find the same controlled combination: an explicit metric ground-plane disc whose direct effect is evaluated at four public radii, with critical-radius search confined to the private oracle; true simulator re-rollouts; reset-matched do(action)/do(body)/do(sensor); three-layer ground truth separating physical fact, future observation, and visual support; and a preregistered operational visible-support publication contract.

**措辞红线**：禁用"首个 / 没有先例 / 没有评测回答过 / all mainstream benchmarks lack a
body"。贡献表述为 "We explicitly operationalize …"。

**Claim–evidence 红线四条（v1.3.2 冻结；§0/Introduction/Limitations 三处一致）**：

| 维度 | 不能主张 | 可以主张 |
|---|---|---|
| General world model | "the first systematic VLM world-model benchmark" | a narrower behavioral capability: visible-space, short-horizon action-consequence prediction under explicit body and sensor conditions |
| Structured state transition | "the first structured state-transition evaluation for VLMs" | terminal ego-state mechanically derived from true simulator re-rollouts under metric action/body/sensor interventions（非语言/标注 trace） |
| Body-aware evaluation | "existing benchmarks ignore embodiment" | 差异 = controlled conjunction：单张标定 ego 图 + Direct Body Effect 每题四个公开平面圆盘半径（临界半径搜索仅在 private oracle）+ metric action program + reset-matched true re-rollout |
| 能力外推 | 由结果推出内部完整 WM / 长时规划 / 策略排序 / 一般动力学理解 | outcome fidelity、reset-matched action fidelity、planar-footprint body responsiveness、sensor invariance/responsiveness、terminal consequence projection（§0 证据边界句） |

---

## 4. 三层真值与可视发布契约

### 4.1 三层 GT（每条 rollout 分开存，互不替代）

| 层 | 含义 | do(sensor) 能否改变 | v1.1 角色 |
|---|---|---|---|
| **Physical GT** | 碰撞、停止点、终点、目标距离/方位、物理探针安全 | 否 | 公开答案键 |
| **Observation GT** | 初始可见目标在未来相机中的可见性/投影 | 是（正确地变） | Q8 答案键 |
| **Evidence GT** | 初始公开图像能否支持该答案 | 是 | **私有发布门**（不再是答案选项） |

### 4.2 非对称可视发布契约

证据要求不对称（"存在障碍"是局部正证据，"不存在障碍"是接近全程的否定证明）：

| 物理 GT | 发布要求 | 处置 |
|---|---|---|
| collision | 接触障碍可见可定位；起点到接触点的扫掠**前缀**被支持；归因不依赖被遮挡几何 | 发布 |
| safe | 整条扫掠走廊过预注册覆盖门；无关键 out-of-FOV / occlusion / invalid depth；`max_unsupported_run_m` 达标；对证据阈值扰动稳定（**操作性否定证据**，不宣称数学证明） | 发布 |
| 目标视线（Q8 及一切 target 类投影） | 目标初始可见且唯一；对应关系跨 profile 稳定；**决定未来可见性的关键遮挡表面须在初始图中被支持**；未来可见性依赖完全不可见的新区域 → 不发布 | 发布 / 不发布 |
| 任一，证据门失败 | — | **不发布**。物理 GT 私有保留，进审计与 yield 统计 |

覆盖阈值附近设不发布灰区并做敏感性分析。私有 full geometry 用于防止隐藏几何污染标签。

### 4.3 Sufficient-only 发布规则（v1.1，替代旧弃权规则）

- 公开 closed 选项中**不存在** `insufficient_evidence`；open schema 中**不存在**
  `answerable` 字段——每道发布题都满足可视支持契约，模型必须作答（Physion 式
  forced choice）；人类可答性由 §8.6 human gate 按 head × variant 验证；
- 题面协议改写为："每道题都保证可以从可见空间作答"；删除弃权指引；
- 各 family 的随机基线因此简化（纯任务选项数决定），null 值预注册（§8.2）；
- **代价须审计**：sufficient-only 筛选与场景形态相关（safe 需开阔视野、collision 只需
  局部可见），可能让图像表面统计与物理标签产生选择性相关 → §9.3 的选择偏置审计
  是本契约的强制配套。

---

## 5. 分类体系：二维 + 一属性

**为什么不按难度分类**：VSI-Bench/BEAR 按被测能力性质组织任务；我们旧的 L0–L4 与
WM 阶梯的 L0–L7 撞名；Lost in Aggregation 的不单调实测是支持性佐证。难度降为分析属性。

### 5.1 行：四个原子结果头 + 一个整合 track（按 GT 语义划分）

划分判据：**同一行的 GT 在 do(sensor) 下行为必须一致**。

| 结果头（发布名） | 任务（内部编号） | GT 类型 | do(sensor) 后 |
|---|---|---|---|
| **Interaction** | 是否碰撞(Q1)、停止段(Q2)、接触物(Q3) | 物理 | 不变 |
| **Endpoint & Target Relation** | 目标 range 趋势(Q6)、终点方位(Q7)、终点位姿诊断（translation/heading，不进 headline） | 物理（世界坐标） | 不变 |
| **Future Observation** | 初始可见目标的未来可见性(Q8) | 传感器相对 | 正确地变 |
| **Local Affordance** | 终点固定五探针物理安全掩码(Q9) | 物理 | 不变 |
| **Constraint Integration**（独立 track） | 多约束候选集合选择(Q10) | 组合 | 单列 |

诊断（不进任何总分）：Q0a 当前距离（尺度锚定；保留在公开 diagnostic track，评测须
stateless，同场景 Q0a 答案不得进入后续题上下文）、Q8 leaving-count。
Q6 geodesic_progress 是 Endpoint & Target Relation 的 closed-only variant（非独立头），仅
HM3D/R2R（GS 发布即 release failure；审计实测 GS 原始可用率 0/2378，HM3D 89.4%、
R2R 97.2%）。状态=诊断，直至 navmesh-vs-surface 冲突来源审计通过后升为计分 variant；
冲突样本只进独立 divergence 诊断集，永不并入主分（§13.7）。

### 5.2 列：四种干预协议

| 干预 | 固定 | 改变 | 预期行为 | 指标 |
|---|---|---|---|---|
| **Information ablation** | RGB、物理状态、GT | 四条件：full / no-height / no-FOV / no-height-and-no-FOV | 掉分=字段有用；height 与 FOV 的效用**独立可归因** | 逐条件 paired Δ |
| **do(action)** | 图像、pose、身体、sensor | reset-matched 动作对（§6.7） | 物理后果**正确地变** | pair exact + effect-transition 正确率 |
| **do(body)** | 图像、动作、世界 | 圆盘半径（真实重 rollout） | 直接接触与下游状态**正确地变** | direct effect + propagation（§7） |
| **do(sensor · height)** | 世界、pose、动作、身体、HFOV/VFOV | 真实相机**高度**（重渲染；视点位置与遮挡改变） | 物理**不变**；view **正确响应** | 按因子分报（§8.4） |
| **do(sensor · FOV)** | 世界、pose、动作、身体、相机高度 | **HFOV/VFOV**（重渲染；仅投影视锥改变） | 物理**不变**；view **正确响应** | 按因子分报（§8.4） |

Information ablation 沿用相同物理 item；隐藏标定后输入可能尺度欠定——该条件测
**公开标定信息的效用**，不宣称 ablated 输入仍满足完整可视支持契约。

### 5.3 稀疏因子矩阵（首版覆盖）

| | absolute | do(action) | do(body) | do(sensor) | info ablation |
|---|---|---|---|---|---|
| Interaction | 正式 | 正式 | **正式** | 物理不变性 | 正式 |
| Endpoint & Target Relation | 正式 | 正式 | **正式候选（propagation，pilot 定 Tier）** | 物理不变性 | 正式 |
| Future Observation | 正式 | 正式 | pilot | 视图响应性 | 正式 |
| Local Affordance | 正式 | 正式 | pilot | 物理不变性 | 正式 |
| Integration | 单列 | 有 | 后续版本 | 仅纯物理 profile（§6.6） | 诊断 |

正交性要求：行只描述预测对象、列只描述被改变量、每个论文主张指明支撑格子、
body 不同时出现在两轴。不要求填满。

### 5.4 对 2026-07-25 freeze 的正式修订（四处）

1. **`body-counterfactual` 不再是 capability 行**。Q4 归 do(body) 列的 Direct Body
   Effect；它是 **R1 主张的必报主结果**（required primary intervention result，
   出现在摘要、主结果表和 claim matrix），不是普通 result-card 切片（§8.1）。
2. **Q10 移出 `maneuverability`**，单列为 Constraint Integration（BEAR 把原子能力类与
   Long-horizon integration 分开组织，是此结构的先例）。
3. **headline 为四原子结果头的宏平均；Constraint Integration 为独立 track，不参与
   headline**（four-head headline plus an independent Integration track）；
   原 `relation`/`view` 拆分保留，改发布名。
4. **（v1.1，用户决定）删除 Evidence Responsiveness headline 与 R3 主张**。
   `insufficient_evidence` 移出公开选项；Q5b 取消；Q5a 保留为 Sensor Physical
   Invariance。7/25 P0-9 的双 headline 条款被本条 supersede。

### 5.5 属性（非评分）

`rollout_stage = 0..4`；**`execution_regime ∈ {completed_clear, contact_truncated,
invalid_geometry}`（v1.3.3）——非评分属性但承担发布路由：结果头回答"测什么"，
execution regime 回答"动作怎样结束"；core 发布资格由它决定（§12.4 门 1），validator
从私有 rollout 独立重算；**该字段是信任边界：只存于 private answer，公开 item 出现
即 violation（公开 contact_truncated = 直接泄漏 Q1/Q2 答案），trusted-local review
模式下才显示**；另记 `primitive_count /
forward_leg_count / counterfactual_branches / constraint_count /
requires_translation / requires_viewpoint_update`。单调性留待实证。

---

## 6. 任务与题型设计（含真实例子）

例子来源：〔真〕= `candidate_qa_q4_protocol_v7_with_cases` 实际产物；
〔构〕= 真实 builder 在合成 fixture 上渲染。注意：例子取自 v1.0 产物，其 closed 选项
仍含 `insufficient_evidence`；v1.1 编译时按 §4.3 移除。

每道题 = 一张 RGB + 固定前缀（相机高、HFOV/VFOV、身体半径或候选半径、认证近地带长度）+ 问题。

### 6.0 底物单位与目标绑定（v1.3，validator 强制）

**Target 定义（五条）**：初始 RGB 可见；唯一物理 `target_instance_id`；具体类别
（chair/sofa/table 级，排除 misc/unknown）；初始像素面积与地面支撑点达门槛；
**同一 intervention group（action/body/sensor 全部干预）内恒指同一物理实例**。
题面只写 "the unique visible chair"，instance id 私有。

**绑定不变量（v1.3.1 两层）**：Q1/Q2/Q3/Q9 绑定 `base_rollout_key`（目标无关，
**不因目标重复生成**）；Q6/Q7/Q8 绑定 `target_projection_key`；joint-vector 与 Body
Propagation = base + 一个固定目标；Q10 = 多候选动作 + 固定目标。同一条链以私有
`target_group_id` 关联，validator 校验两层键与目标参考集 identity——不允许 Q6 问椅子、
Q7 问桌子，再宣称测到了同一条未来状态链。

**一个完整链例**〔设计例〕：唯一可见 chair，r=0.15 m，动作 "F0.5 → R45° → F0.5"，
私有 GT `{collision: false, executed_forward_m: [0.5, 0.5], target_range: 3.20→2.64,
target_bearing_after: −56°, target_visible_after: false (cause: out_of_fov),
safe_mask: [T,F,T,F,F]}` → 同一向量投影出六道题：不撞 / 全程执行 / 更近 / 左侧 /
不可见 / 探针 F 与 L15 安全。因果链清晰：动作 → 碰撞 → 停点 → 距离方位 → 可见性 →
可行性。

### 6.1 Interaction

**Q1 是否碰撞**〔真, r2r〕：高 1.65 m，HFOV 110°，r=0.2 m，"move forward 0.5 m" →
`no_collision`（clearance 0.464 m ≥ 门 0.30 m）。

**Q2 停止段**〔真, r2r〕："F1.5 → R45° → F3" → `contact_during_forward_2`；
open：`{contact_forward_leg_number: 2, distance_into_contact_leg_m: 1.894,
total_executed_forward_m: 3.394}`。门：≥2 段 Forward；full 与 depth 接触段一致。

**Q3 接触物**〔真, hm3d〕：**closed primary 用固定二元接触 superclass 词表**
（v1.3.2：`structural_contact / object_contact`，标签对称、固定大小）
→ `structural_contact`。exact contact category（此例中的
`mirror` / `side table` / `umbrella stand` 等实例类别）对每个合格 item 一律保留为
open diagnostic；升格条件见 §6.8。门：碰撞已归因、双 oracle 同实例、正确 superclass
可由已归因的精确类别确定；object contact 的精确类别必须在初始图中可见。closed primary
不再要求动态 exact-category 干扰项。

发布 margin（三题共享）：碰撞后剩余路径 ≥0.50 m；安全间隙 ≥0.30 m。

### 6.2 Endpoint & Target Relation

**Q6 表面距离趋势**〔真, r2r〕：目标 seating，"F0.5 → R45° → F0.5" → `closer`
（6.764 → 6.259 m）。GT 纯符号；微小变化可被 margin 扣留但不得改写成 unchanged。
反捷径门：转后前进 ≥0.5 m 或初始方位 ≥20°。

**Q7 终点方位**〔真, hm3d〕：目标 cabinet，"F2.5 → L30° → F2 → R30°" → `left`
（bearing −93.2°；parallax 83.8° ≥ 门 10°——净转向为 0，答案几乎全由平移造成）。

**执行状态前提（v1.3.3）**：Q6/Q7 primary 只发布 `completed_clear` rollout——无碰撞、
动作完整执行，realized = nominal，读出不与 Q1/Q2 复合；`contact_truncated` 的距离/
方位读出移入 §6.10。**Q6 closed 收为 closer / farther 二分类**：GT 严格按
`target_range` 差值符号（Δd < 0 = closer，Δd > 0 = farther）；"近了 0.4 m" 永远是
closer，不因容差改写；unchanged 在纯符号 GT 下仅精确零可达、属死选项，移出公开选项集
（精确零/纯旋转样本只作限量 diagnostic control）；灰区照旧只决定是否发布。定向采集须
保证 toward / lateral / away 动作使两标签在每 backend 有独立 scene 支持（§13.9-8）。

**距离语义三量（v1.3 冻结）**——"离目标多远"必须指明是哪一个量：

| 量 | 定义 | 用途 |
|---|---|---|
| `target_range_m` | 圆盘**中心**到固定目标点集的 2D 最近距离（半径无关） | **Q6 主读出**；与 §13.3 传播 GT 同一定义 |
| `target_clearance_m` | 身体**边缘**到目标表面的间隙（含半径） | body/contact 诊断 |
| `target_camera_depth_m` | 终点相机坐标中的目标 Z | observation 诊断 |

Q6 的 GT 从 surface gap 切换为 `target_range_m` 趋势——同一中心位姿下 surface gap 被
半径机械平移，与 §13.3 禁用 `final_surface_distance_m` 是同一理由；surface gap 保留为
clearance 诊断字段。Q0a 同步改用 `target_range_m`（诊断，语义对齐）。

**固定目标参考集（v1.3.1 实现级定义）**：点集取自**私有完整实例几何**（非任一 sensor
profile 的可见点云）；同 scene 内 instance_id 稳定；对全部 action/body/sensor sibling
**逐点恒等**；以世界坐标（或 canonical pose-local）存储一次；validator 校验
reference-set identity（哈希）。漏斗审计当前用的初始可见质心是 **proxy**，仅供旧数据
可行性分析，不得作为正式 GT。camera depth 不设独立 headline，仅 open/diagnostic。

**终点位姿诊断读出（v1.3 新增，不进 headline）**：`realized_translation_m`、
`realized_heading_change_deg`。作用是误差归因——区分"终点推错了"与"终点对了但目标
关系推错了"；进 Atomic 协议与链式条件分析。

### 6.3 Future Observation

**Q8 未来可见性**〔真, gs〕：目标 table，"F2.5" → `visible`（终点面积比 0.0195，高于
40px 灰带上界）。只追踪**初始已可见**目标；发布须过 §4.2 的目标视线行。

**v1.3 细化**（防退化成方位/FOV 判断）：
- closed 保持二值，报告必须并列 open 的 center error 与 projected-area error；
- 不可见样本按 `invisible_cause ∈ {out_of_fov, occluded}` **分层报告**（私有几何可判：
  投影出视锥 = FOV，锥内被挡 = 遮挡）；
- 两类设计配对按**干预类型**分立（v1.3.1，= 本 head 的必要性过滤器，§2）：
  **FOV 对** = do(sensor) 型（固定世界/pose/action/body/target/物理终点，只变 FOV，
  测视野边界响应）；**遮挡对** = matched do(action) 型（固定初始图/target/body/sensor
  与动作复杂度、终点 bearing 桶，变走廊/终点视线，测遮挡理解）——不得都称 sensor 对；
- 全部 Q8 的 GT 一律取 **realized endpoint**，禁止 nominal planned endpoint；
- **执行状态前提（v1.3.3）**：Q8 primary 只发布 `completed_clear` rollout（此时
  realized = nominal，上一条自然满足）；遮挡对与 FOV 对两端都须 completed_clear；
  `contact_truncated` 的可见性读出移入 §6.10。preview 审计：61% Q8 落在碰撞
  rollout 上，可见性与碰撞推理混杂，是本条的直接依据。

### 6.4 Local Affordance

**Q9 五探针**〔真, hm3d〕：探针集固定（F1m / L15°+F1 / R15°+F1 / L30°+F1 / R30°+F1）。
例中五探针全不安全：`{certified_safe_action_mask: [false×5],
no_safe_probe_continuation: true}`——只描述五个固定局部探针，不能外推到其他动作。
v1.1 下五条探针走廊的证据须全部通过发布门才出题；答案只来自物理掩码。

**Q9 资格门（v1.3.3 收紧，全部同时满足）**：
① 主动作 `collision == false`；② 动作完整执行（executed prefix = 全程）；
③ 终点五探针证据完整、充分、远离发布边界；④ `endpoint_mask ≠ start_mask`
（必要性过滤器——**只在 completed_clear 集合内运行**：无碰撞前提时该过滤器会
系统性选中碰撞终点，preview 174/175 退化即此根因）；⑤ 正式 matched-action 掩码对
两端均 completed_clear 且掩码不同；⑥ forward_safe 两标签、turn_choice 各标签有
跨场景支持。主动作必须安全完成，但被询问的五个探针本身可以 unsafe——那正是被测
内容（"安全到达另一个位置后还能做什么"，而非"撞墙后周围当然都不安全"）。
`contact_truncated` 终点的掩码只进 §6.10。

### 6.5 Constraint Integration

**Q10**〔构〕：约束"不碰撞 + geodesic 距离↓≥30% + 目标清晰可见 + 保留≥2 个安全后续"，
四候选同长度互异 → `["A1"]`；后果表：A2 撞、A3 目标丢失、A4 只剩 1 个安全后续——
hard negative 各违反一条明确约束，且至少一个合法候选。30% 带未公开缓冲带
（≤25% 负例 / ≥35% 正例）。精确集合匹配。

### 6.6 Q10 × do(sensor) 限定

sensor 干预只用**不含 `target_visible`** 的纯物理约束 profile，使正确集合严格
sensor-invariant。

### 6.7 do(action) 配对契约（正式定义）

> do(action) 使用相同 scene、pose、sensor、body、target 下的 reset-matched action pairs。
> 配对控制：总前进距离桶、primitive count、总转角桶、净转角桶、每段距离 profile 及其他
> 预注册 action-complexity 特征；通过转角分配/路径形状变化使扫掠走廊与物理结果不同。

`pair_id` 只存 private；每 pose 的 pair 数可变；不固定 safe/collision 数量；safe 侧仍走
完整走廊发布门。指标是 GT effect transition（safe→collision、closer→farther、
visible→not-visible、mask A→mask B），模型须同时答对两端与变化关系。
配对漏斗：physical pairs → both-side 发布门通过 → label-changing → independent scene support。

### 6.8 题型五原则（v1.1 修订原则 1）

1. **选项统计对 GT 条件独立**：给定公开输入后，选项数、干扰项分布、文本特征和位置
   分布必须对正确标签条件独立；**除保证正确答案在选项集中外**，GT 不得影响可观察的
   选项统计；每个 closed family 必须过 option-only gate。
   - 合规：Q2 按公开动作的 Forward 数生成 stage 选项；Q4 固定三区间；Q10 列公开四候选。
   - **Q3（v1.3.2 保守默认，终局）**：closed primary readout 固定为标签对称、固定
     大小的**接触 superclass 词表**；exact contact category 对每个合格 item 保留为
     open diagnostic。exact 最多经预注册门**单向升格**为 additional scored secondary
     closed variant，升格门（全部满足）：①接触实例归因跨双 oracle 与全部 sibling
     稳定一致；②exact 类别实体初始图中可见且过 Q3 可视支持条件；③backend 间类别
     映射明确、无 `misc/unknown` scored label；④exact variant 的 human gate
     （paired/absolute）显著高于 null；⑤option-only / category-frequency /
     position-only 攻击全部不过 majority/UCB 门；⑥每个 scored category 有独立 scene
     支持、无单场景垄断；⑦distractor 生成不依赖正确类别的可观察统计。
     **任何门通过都不替换、不重定义 superclass primary readout**。理由：primary
     construct 不随 pilot 结果改变语义；审计"未检出"共现捷径不证明其不存在；exact
     混入物体识别、标签粒度与跨 backend taxonomy 争议。
2. 每道 closed 只有一个答案键。
3. 连续 GT 与发布门分离（符号定 GT，margin 只决定是否发布）。
4. 控制样本走 open：all-safe / all-collision 向量、same-distance、单调性违反。
5. 边界余量 + 灰区逐任务（margins 全部落库）。

### 6.9 三提问协议

| 协议 | 形式 | 测什么 |
|---|---|---|
| Atomic-minimal | 每字段独立、最小题面 | 原子能力 |
| Atomic-context | 独立提问但给完整字段清单 | 任务上下文提示效应（含 schema 与任务先验泄露） |
| Joint-vector | 一次返回完整 consequence vector | 联合输出效应（含共享推断、自洽约束、错误传播） |

v1.1 下发布集全部 sufficient，三协议天然共享同一 item 集合；仍须相同解码参数。
差值只作行为描述，不宣称恢复内部推理。字段顺序与 schema 固定或随机化审计。

### 6.10 Post-contact Consequence Propagation（v1.3.3 新增，诊断 track）

碰撞截断后的下游事实（截断点处的目标距离、可见性、局部空间）仍是有效物理 GT，
且直接体现状态传播能力——但它是 **动作 → 接触 → 提前停止 → realized endpoint →
下游关系** 的显式链式任务，不得伪装成原子 Q6/Q8/Q9（否则：答案空间被 GT 动态决定、
距离推理与碰撞识别混成一个分数、识别碰撞即可绕过距离问题、并与 Q1/Q2/Q3 重复）。
独立成 track：

- **接触是给定条件，不是被测项**：题面告知 contact 发生（隔离传播与检测）；冻结句
  （英文）：*Contact is conditioned on; collision detection is neither requested nor
  scored.* 计分向量不含 `collision` 字段；
- 输出为六字段联合向量（= `VARIANT_OPEN_RESPONSE_FIELDS[("Q2",
  "post_contact_consequence")]` 实现）：`contact_forward_leg_number /
  contact_category / total_executed_forward_m / target_range_change_m /
  target_visible_at_stop / terminal_physical_safe_mask`；
- 计分用 joint exact 或分阶段条件分数，明确声明链式构念；
- item 标 `score_role = post_contact_diagnostic`，不进四头 headline、不进 VCS；
- 初版为 diagnostic；支持量、人类可答性、捷径门全部通过后方可**单向升格**为正式
  结果（类比 Q3 exact 升格门，通过不改变四头 primary 语义）；
- 与 §7.2 Body Propagation 正交：后者是 do(body) 的跨半径链式效应，按 Tier 规则
  发布，不受本节影响。

---

## 7. Body 主张：两级证据 + 发布分级

### 7.1 Direct Body Effect（现有 Q4，冻结；R1 必报主结果）

同图同动作四递增半径。Closed：三个相邻 transition 区间（三分类、无弃权项，
transition-rank macro 把常量预测器锁死 1/3）；Open：四元 safe/collision 向量 +
all-safe / all-collision / 单调性控制。真翻转要求 `transition_rank == len(safe)` 且
r\* 在灰区外。公开输入仅 `candidate_body_radii_m`；near-field 用最大候选。

〔构〕例：radii [0.15, 0.2, 0.3, 0.45]，"R30° → F1.5" → `between_2_3`
（vector [safe, safe, collision, collision]，r\*=0.25）。
〔真〕控制例：[0.18, 0.225, 0.315, 0.5]，"F0.5" → 全 safe（仅 open）。

### 7.2 Body Propagation（与 flip 解耦）

构念：`do(radius) → Δendpoint → Δtarget relation`。

主问题〔设计例〕：
> For the two body radii executing the same action, which execution ends closer to the
> visible target? — smaller body / larger body（primary 不含 same；same 是限量控制）

**传播 GT（§13.3 裁决，冻结）**：`d_i = dist_2D(圆盘中心 @ realized endpoint_i, 固定
目标参考集)`——目标参考集 = 该 record 一次性存储的目标地面支撑点集（与半径无关），
主读出取中心到该点集的最近点距离；`Δd = d(r_l) − d(r_s)`。**禁用**
`final_surface_distance_m`：它减掉了身体半径，同一中心位姿下大半径机械性少
(r_l − r_s)，是构造伪影不是传播。旧记录未持久化目标点集，只能做质心代理分析，
不构成正式标注；新采集必须持久化目标点集与每半径的物理接触 instance id。

| stratum | 两身体状态 | 传播来源 | 证据成本 | 地位 |
|---|---|---|---|---|
| flip-based | 一 safe 一 collision | 是否完成动作变化 | 贵（safe 侧走廊否定证明） | 正式候选 |
| BC-simple | 都碰撞，同段同接触源，Δstop≈r_l−r_s | 简单提前接触 | 便宜 | **仅诊断**（防"大的更早停"常识题虚增分数） |
| BC-branching | 都碰撞，不同段/不同接触源/终点差显著放大 | 执行分支改变 | 便宜 | **主候选** |

逐组落库：`contact_leg_{small,large}`、`contact_source_{small,large}`（用稳定的物理
instance/source id，不用可能抖动的语义类别）、`endpoint_delta_m`（**realized endpoint
的地面平面欧氏位移**）、`endpoint_delta_m / |r_l − r_s|`（放大率）、
`target_distance_delta_m`。放大门数值不在审计前冻结。

Primary 六门（v1.1 修订第 1 门措辞，避免误杀 both-collide）：

1. **至少两个半径的 realized execution state 不同**——executed prefix、contact
   leg/source、realized endpoint 至少一项不同（不要求 safe/collision 标签翻转）；
2. 至少一个身体提前停止；
3. 终点差过位姿灰区；
4. |Δd| 过度量灰区；
5. 当前 profile 下两身体的发布门都通过；
6. **large-closer / large-farther 双向场景支持**（独立 blocking 项）。

方向按干预前 nominal displacement 与目标方向点积分 toward/away/lateral；定向采集对
**away + 大身体早停**（large-closer 的稀缺来源）单设搜索预算。

### 7.3 发布三级（预注册，禁止事后放宽）

| 级 | 条件 | 论文主张 |
|---|---|---|
| **Broad Tier A** | flip-based 与 BC-branching 都有独立 scene/标签/human 支持 | 跨两种机制的 body propagation |
| **Stratum-specific Tier A** | 某一 stratum 单独达标 | 发布该 stratum 名称明确的分数 |
| **Tier B** | 只有 BC-simple / 标签单边 / 支持不足 | 探索性诊断进结果卡；论文只主张 direct body effect |

---

## 8. 评分体系

### 8.1 Claim matrix（v1.1：单 headline + 必报主结果）

| 主张 | 支撑分数 | 地位 |
|---|---|---|
| **R1 body-conditioned** | **Direct Body Effect**（+ Body Propagation 按 Tier 限定措辞） | **required primary intervention result**：摘要、主结果表、claim matrix 必报 |
| **R2 visible consequence** | **Visible Consequence Score**（headline）+ Action Fidelity | headline + required primary |
| ~~R3 evidence~~ | —（v1.1 移出范围） | — |
| Integration | Constraint Integration | 独立组合结果，单列 |

唯一 headline：**Visible Consequence Score** = 四结果头的层次宏平均
（closed ∧ 发布门通过 ∧ primary_balanced）。
Q1–Q3 题面印半径但不做配对 do(body) 时模型可忽略半径答对，**故 R1 不得只由
Visible Consequence 支撑**——这是 Direct Body Effect 必报的原因。

### 8.2 Null-corrected 聚合（v1.1 新增，解决各任务 chance 不同）

原始准确率直接宏平均会让高 chance 任务天然占优。校正在 **item 级**进行——选项数可随
公开输入变化（如 Q2 的 stage 选项数随 Forward 段数变），null 是 item-specific 的：

```
s_adj_i = (score_i − null_i) / (1 − null_i)      # item 级；允许负值，不裁 0
```

`null_i` 由该 item 的**预注册随机策略**决定（选项均匀 / 集合任务的预注册随机集合
策略；v1.1 无弃权项，null 只由任务选项结构决定）。然后按
`s_adj_i: item → intervention group → scene → variant → atomic task → outcome head →
四头宏平均`层次聚合。**原始 accuracy 用相同权重并行聚合、全程并报**。

### 8.3 切片（结果卡）

Action Fidelity ／ Direct Body Effect ／ Body Propagation（分 stratum）／
Sensor Physical Invariance（含 Q5a）／ Sensor View Responsiveness ／
Calibration Utility ／ Constraint Integration ／ Oracle Decomposition（§8.8，v1.3.2）／
Input-Binding Corruption（§9.2，v1.3.2）／ Post-contact Propagation（§6.10，v1.3.3）
——后三者不进任何 headline 或 required primary score。

### 8.4 Sensor 指标（v1.1 两个，Q5b 已取消）

- **Sensor Physical Invariance（Q5a）**：两 profile 都过发布门时，物理答案正确且一致。
  聚合：`pair exact → physical group → scene → physical outcome head →
  required-head macro`；required heads 至少为 **Interaction、Endpoint & Target Relation、
  Local Affordance**；Q8 不进 invariance（其 Observation GT 随 sensor 正确变化，
  单独归 View Responsiveness）；每 head 预注册最低 scene/group 支持；
  缺 head **不得静默重归一化**（沿用 missing-family policy）。
- **Sensor View Responsiveness**：初始可见目标的未来可见性随 profile 变化时预测正确
  地变。
- 两指标一律按 **height-only** 与 **FOV-only** 因子分别报告（高度改变视点位置与遮挡，
  FOV 只改变投影视锥），不得合并为单一 sensor 数。

### 8.5 统计纪律

- scene-cluster bootstrap CI；同一 item 可同时进能力分与干预切片，但**跨分数比较必须
  在同一次 bootstrap 重采样中直接算差值 CI**，禁止目测两个边际 CI；
  覆盖场景不一致时主比较限制在共同 eligible scenes；
- 各分数量纲不同不可横比须显式声明（S_adj 缓解但不消除——集合任务与分类任务仍不同构）；
- 另报 family-macro 与 item-micro 验证排名是否依赖聚合权重。

### 8.6 人类基线与泄漏诊断

- 人机**同刺激**（Physion 范式），**按 head × variant 分别报告**，不做混合 human
  score；盲答式 + 错误双向追溯（VSI-Bench 范式）；human answerability gate 同样按
  head × variant 检查——它是"可答"的最终验证，几何契约只是操作性筛选（§0）；
- **人类 do(action) 配对检验是 construct-validity gate**（§9.2）；
- Human-normalized 仅在 `LCB(Human) > UCB(Null)+ε` 且 Human−Null 超预注册最小分母时
  定义，否则标 undefined；
- 另报 **within-to-cross-scene generalization gap**（仅有额外污染证据时才称 leakage）；正式榜只用 scene-atomic cross-scene split。

### 8.7 难度分布：先报告，不设硬门

| Head | 难度变量 |
|---|---|
| Interaction | min clearance、contact arc、动作长度 |
| Endpoint & Target Relation | endpoint displacement、parallax、distance delta |
| Future Observation | target pixel area、边缘距离、occlusion fraction |
| Local Affordance | safe probe count、probe margin |
| Body Direct | critical radius、transition rank |
| Body Propagation | endpoint delta、distance delta、flip/BC subtype |

compiler 输出分布报告；发布要求分布透明、关键区间有 scene 支持；硬分桶等 pilot 后定。

### 8.8 Oracle 分解梯子与公共几何管线基线（v1.3.2 新增，诊断族）

目的：构造显式误差定位链 **RGB metric perception → action/body rollout →
realized endpoint → target/view/affordance update**，把总缺口分解为可归因的阶段缺口
（RQ：错误来自单目度量几何、动作 rollout、终点更新，还是下游投影/组合）。

| 条件 | 使用信息 | 适用头 | 地位 |
|---|---|---|---|
| **Public Geometric Pipeline** | 公开 RGB + calibration + radius + metric action；公开模型估 metric depth；target 头用公开 open-vocabulary grounding | 四头 | 公平 baseline（仅公开输入，可进主结果表） |
| **Oracle-Depth Pipeline** | 同上管线，深度换 simulator 初始 metric depth | 四头 | privileged diagnostic |
| **Oracle-Endpoint** | 提供真实 `realized_endpoint`（必要时含真实 executed prefix），其余推理不变 | ETR / Future Observation / Local Affordance；**Interaction 整头不适用（`—`）** | privileged diagnostic |
| **Full-Geometry Oracle** | 私有完整几何 + 真实 target instance/参考集 + 真实 body/action/sensor | 四头 | sanity ceiling（GT/margin/投影/evaluator 自洽检验），非排行榜条目 |

冻结规则：

- **Interaction 在 Oracle-Endpoint 下一律 `—`**：终点/已执行前缀直接泄露是否提前
  停止、哪个 forward leg 未走完与接触大致位置——Q1/Q2 直接泄露、Q3 间接泄露，
  故按整头屏蔽而非仅 Q1。
- **基线防泄漏**：Public Geometric Pipeline 禁用 GT target mask / GT 目标参考集。
  target grounding 分三档并落库标档：public grounding（公开 RGB + 题面目标描述）/
  oracle grounding（privileged diagnostic）/ full geometry（仅 sanity ceiling）——
  防止 "Monocular Depth + Analytic Rollout" 在 Q6/Q7/Q8 上悄悄变成
  "GT 目标实例 + predicted depth"。
- **解释纪律**：报告 **paired diagnostic gains**——
  `Δ_depth = S(oracle depth) − S(predicted depth)`、
  `Δ_endpoint = S(oracle endpoint) − S(full public input)`；二者模型路径不同，
  **禁止机械相加**为误差的严格可加分解。统计沿 §8.5：共同 eligible scenes、
  同一次 scene-cluster bootstrap 内直接算差值 CI。
- 全部条件为诊断族：不改构念、不进 headline、不作 release gate；Public Geometric
  Pipeline 因只用公开输入，作为对照基线可进主结果表。

---

## 9. 反捷径审计体系（v1.1：目标全部是物理标签）

### 9.1 现有物理侧攻击（已实现：`pipeline/shortcut_audit.py`）

7 攻击（action_only / total_distance_threshold / same_image_group_rank /
body_radius_only / radius_options_only / option_only / backend_family_only）；
场景聚类配对 bootstrap；`UCB(attack − majority) ≤ 0.05`；pass / fail /
**inconclusive ≠ pass**；scene-disjoint 与 group-disjoint 双 folds。
**v1.1 补充**：审计必须在 **family × variant 内**进行（跨 family 混合会稀释信号）。

### 9.2 Action-conditioning 检验（v1.1 修正：即正式 do(action) 配对，无 "wrong action" 概念）

换入另一动作后它就是**另一道有自己 GT 的合法问题**——按原 GT 打分是 prompt
corruption，不是 action-conditioning。有效检验就是 §6.7 的配对本身：
同图 + action A → GT A；同图 + matched action B → GT B（匹配复杂度、GT 不同）。

| 检验 | 定义 | 地位 |
|---|---|---|
| 人类 construct-validity gate | 按 head × variant：配对两端 paired exact 显著高于 null；GT effect transition 判断显著高于 null；label-changing pair 的变化方向与 GT 一致 | **数据集门**：不过则该 head × variant 回炉 |
| 模型 do(action) 配对 | 同一协议 | **正式结果**（= Action Fidelity），不作为数据集 release gate（模型失败不证明数据集无效） |
| **Input-Binding Diagnostics（v1.3.2 常设强制诊断）** | 见下方冻结定义 | 强制诊断；**不是** do(action)/do(body)：不进 Action Fidelity、不作 construct-validity 门、不作 release gate |

**Input-Binding Diagnostics（v1.3.2 冻结定义）**——正式名 **Action / Body
Prompt-Binding Corruption**（表格/论文中禁称 shuffled-action / shuffled-body
*intervention*）：人为破坏当前题面中的 action 文本或半径数字、保留原 item 为参照，
检验模型输出是否**行为上绑定**对应公开输入字段。与 do() 的边界一句话：
do() = 新输入 + 新 GT + 真实重 rollout；corruption = 无新 GT 的题面破坏。

- **采样纪律**：action donor 取同 family × variant，并匹配 primitive 数 /
  forward-leg 数 / 总前进距离桶 / 总与净转角桶 / 文本格式与近似 token 长度；
  body corruption 取合法范围内另一半径，匹配数字格式、小数位、radius 桶与候选集
  结构——防止响应只来自"文本明显变长 / 数字分布异常"而非字段绑定。
- **可报指标**：Prediction Change Rate（改变 ≠ 改对，随机不稳定也会推高）；
  Original-Answer Confidence Drop（仅 logprob 可得的模型）；Full-to-Corrupt Paired
  Score Drop（只能称 corruption drop，不得解释为物理正确性）；Corruption
  Consistency by Head（哪些 head 更忽略 action/body 字段）。
- **不可报**：physical accuracy、effect-transition correctness、
  construct-validity、release gate——corruption 后的混合题面**没有真实新 rollout**。
- 冻结解释句（英文）：

  > A response change under prompt corruption is neither necessary nor sufficient
  > evidence of physically correct action/body conditioning. These diagnostics only
  > test whether model outputs are behaviorally bound to the corresponding public
  > input fields. Construct evidence is provided exclusively by reset-matched
  > re-rollout pairs with sibling-specific ground truth.

### 9.3 选择偏置审计（sufficient-only 契约的强制配套；v1.1 修正触发条件）

sufficient-only 发布使"哪些题被发布"与场景形态相关（safe 需开阔视野、collision 只需
局部可见）。但 **RGB 能预测碰撞可能是合法的局部几何理解**——不能因"图像整体可预测"
触发再平衡，否则模型越会看图、数据越被重平衡掉。选择偏置定义为**筛选造成的额外捷径**：

- 同一 nuisance baseline 分别在 **raw candidate pool** 与 **sufficient-only artifact**
  上训练与评估，报告**筛选前后的增益差**（这才是筛选引入的部分）；
- **blocking** 只覆盖 action-only、metadata（backend/profile identity）和
  artifact（模板/选项/题长）probe；
- 浅层公开 RGB 统计（亮度/对比度/边缘/模糊）、冻结视觉编码器 probe、公开 RGB
  估深和局部几何特征只作**能力诊断**，公开报告，**永不触发再平衡**；
- matched rebalance 只针对已识别的 nuisance feature，不针对整体 RGB 可预测性；
- metadata / artifact / action-only → 物理标签：**blocking**（near-chance，§9.1）。

私有 rendered depth、真实 coverage、failure counters 只作 privileged diagnostic，
不得混入公平基线。方法论目标：

> 物理标签必须由"图像中的**局部几何** × 当前动作/身体"决定，不能由全局图像质量、
> 相机 profile 或题面身份直接猜出——而局部几何信号本身是被测能力，不是 nuisance。

---

## 10. 采集 pipeline（冻结数据流）

```
pose
→ 固定 reference profile 做相机/地面标定
→ 仅【采集有效性门】：canonical floor 拟合质量 / navmesh 与 support validity /
  pose clearance / 渲染完好
→ action/body 的 full-geometry 搜索
  （含 propagation 定向搜索：away + 大身体早停 单设预算）
→ 为物理 survivor 渲染【全部】sensor siblings
→ 每 profile 对称计算 observation GT 与 evidence GT
→ compiler 按 profile 组合建组：
    某 profile 发布门通过        → 该 profile 的核心 consequence item
    两 profile 都通过            → Q5a physical-invariance pair
    任一 profile 未通过          → 该侧不发布；组保留供审计与 yield 统计
→ 灰区/资格/模板/平衡/选项构造 全在 compiler
→ scene-atomic + intervention-group-atomic split
→ shortcut / human / manual-GT / release gates
```

**早期禁止的门**（任务证据早筛）：action corridor coverage、target evidence、
terminal probe evidence、最终 evidence_status、任何由任务与 sensor profile 共同决定的
证据条件。理由（v1.1）：证据随 profile 变化，reference-profile 早筛会丢弃"他侧 profile
本可发布"的组（纯 yield 损失），并使发布集与 profile 产生选择相关（§9.3 要审的偏置）。

**reference 偏差审计**：canonical visible-floor ratio 等标定量仍来自固定参考相机，
只作采集有效性门；必须报告 reference floor-calibration quality 与最终 per-profile
发布状态的关系。

**标签平衡（v1.1 措辞修正）**：candidate pool 保留自然分布；artifact compiler 在
family × variant 内**跨 pose** 做预注册平衡。不固定每 pose 的 safe/collision 数量
（沿用 P0-1/P0-2）；存连续 GT 不提前离散化。

**Target eligibility 与 head 发布门分开**（Q6/Q7/Q8 的 sensor 配对）：
- Target-reference eligibility：两 profile 下同一 private target instance、公共语言
  唯一指认、可见面积/支持点足够、描述不随 profile 歧义；
- Head 发布门：目标可指认后，距离/方位/未来视图是否可认证。
一个 profile 中目标不可指认的组不产生 Q5a 配对（进独立诊断，不伪装成配对）。

---

## 11. 统一漏斗审计（第一个实验）

**执行顺序（v1.1 采纳）**：先对**全部现有连续 record**（不只 compiler 过滤后的 preview）
跑漏斗脚本——验证审计逻辑并清点 both-collide 存量；再按缺失 stratum 做小规模定向
pilot；同一脚本复跑。

审计要做**三个正式决定**：Body Propagation 以哪一级发布（Broad / Stratum-specific /
Tier B）；各 head 在 sufficient-only 契约下的可发布 yield 是否足够（含 Q6 geodesic
去留）；sensor 配对是否存在 reference-profile 或形态选择偏置。

```
physical groups
→ four-radius rollout complete

Direct Body Effect:
→ direct flip → monotonic transition → transition-rank support

Body Propagation:
→ flip-based candidates
→ both-collide candidates（BC-simple / BC-branching 分开）
→ endpoint delta 过灰区 → target-distance delta 过灰区
→ toward / away / lateral（flip 与 BC 分开报）
→ large-closer / large-farther scene support

Sufficient-only yield:
→ 每 head 的发布门通过率（按 backend）
→ Q6 geodesic 可发布率
→ reference-profile 未通过但他 profile 通过的比例
→ both-profile target-reference eligible
→ both-profile head 发布门通过
→ both-sufficient Q5a pairs

选择偏置预检:
→ 物理标签 × nuisance 变量（backend/profile/亮度/模糊/模板/全局统计）
→ 同一 baseline 在 raw pool vs 发布集上的增益差
→ 发布/未发布 × 场景形态统计
```

计数一律按 `backend × unique scene × unique physical group`；
sensor siblings 只贡献配对信息，不增加物理有效样本量。
已知现状：preview 池 direct flip = 0（18 组 Q4 全为 all-safe 控制），r\* 分布
n_eff = 26 frame / 22 scene——flip 级大概率断，定向采集是主路径；
both-collide 存量是本审计最重要的未知数。

---

## 12. 冻结状态与现状差距

### 12.1 状态（Design Contract v1.3.1a）

| 模块 | 状态 |
|---|---|
| 构念、三层 GT、非对称可视发布契约、sufficient-only 规则 | 冻结 |
| 四结果头 + Integration track × 四干预列（含 5.4 四处正式修订） | 冻结 |
| 单 headline + claim matrix（R1/R2；Direct Body Effect 必报主结果） | 冻结 |
| Null-corrected 聚合 + 原始分并报 | 冻结 |
| Direct Body Effect（Q4 现行设计） | 冻结 |
| Propagation 分 flip / BC-simple / BC-branching + 三级发布 + 修订门 1 | 冻结 |
| do(action) matching contract | 冻结 |
| Sensor per-profile qualification（禁任务证据早筛）+ Q5a 聚合与缺失策略 | 冻结 |
| 审计体系（family×variant 内 blocking 门 + 选择偏置诊断 + action-conditioning 对照） | 冻结 |
| 三提问协议 | 冻结 |
| 措辞红线（含 v1.3.2 claim–evidence 四红线、证据边界句） | 冻结 |
| Q3 superclass primary（固定词表 closed readout；exact 永不替换，仅可升格 secondary） | 冻结（v1.3.2） |
| Oracle 分解梯子 + Input-Binding Diagnostics（§8.8 / §9.2，诊断族） | 冻结（v1.3.2；实现差距见 §12.2） |
| Execution regime 路由（Q6–Q9 core 仅 completed_clear；Q9 六门；post-contact 独立 track） | 冻结（v1.3.3） |
| Q6 closed 二分类（closer / farther；unchanged 死选项移除） | 冻结（v1.3.3） |
| 数值（灰区、网格、配额、Tier、BC 放大门）+ 预注册分支选择（Q3 exact secondary 升格资格、Q6 geodesic 升格） | **pilot 依预注册规则决定** |

### 12.2 实现收口状态（2026-07-28）

下表取代早期版本的“待实现”快照。`implemented` 只表示代码路径和确定性测试
已经存在；涉及产量、配额或统计支持的条目仍须由定向 pilot 决定。

| # | 契约项 | 代码状态 | 尚欠证据 |
|---|---|---|---|
| 1 | 四结果头、item-level null、层次聚合与 claim matrix | **implemented**：`scripts/eval_benchmark.py` | 正式 artifact 的各 head scene 支持 |
| 2 | 可变动作数 + 私有 matched-action units，禁止固定 3+3 | **implemented**：`action_sampling.py`、collector、validator | matched pair yield 与 Action Fidelity 人类门 |
| 3 | sufficient-only forced choice；公共 schema 无弃权/`answerable` | **implemented**：builder、validator、公开协议 | 各 head 的 sufficient-only yield 与选择偏置 |
| 4 | Q5a physical invariance + view responsiveness；height/FOV 分因子 | **implemented**：compiler、evaluator | required heads 的 paired-scene 支持；当前 preview 为零 |
| 5 | family×variant 物理捷径门 + raw-vs-published 选择偏置审计 | **implemented**：7 个物理攻击；metadata/artifact blocking；shallow-RGB diagnostic | 冻结 encoder、公开 RGB-depth、局部几何 probe 仍是外部诊断扩展，不得成为再平衡门 |
| 6 | Direct Body + Body Propagation（flip / BC-simple / BC-branching） | **implemented**：builder、scorer、funnel、reserve modes | 当前 preview 无可评分 propagation；须定向采集并判 Tier |
| 7 | 语义 result head + `rollout_stage` 属性；移除旧 L 轴 | **implemented (QA v14)**：artifact、HTTP loader、viewer；公开 `level` 字段为 violation | 无 |
| 8 | Q3 superclass primary + exact secondary 升格门（v1.3.2） | **implemented**：closed 固定为 `structural_contact / object_contact`；exact 类别只进 open；validator 重算固定选项契约 | exact secondary 是否升格仍须在新 raw pool/artifact 上通过七条预注册门 |
| 9 | `target_range`、终点诊断、target binding、Q8 cause、起点掩码 | **implemented**：record、rollout、builder、validator | Q8 两类配对与 Q9 mask-changing pair 的定向配额 |
| 10 | 两层底物键、目标参考集哈希、Q7 答案级反事实、四头 viewer | **implemented**：record、validator、compiler、viewer | 用新采集数据重编 v1.3.1a artifact 并做人工案例审查 |
| 11 | §8.8 oracle 梯子（public geometric pipeline / oracle-depth / oracle-endpoint / full-geometry sanity scorer） | **not implemented** | 四条件 runner + paired-gain 报告（共同 eligible scenes、同 bootstrap 差值 CI）；Oracle-Endpoint 对 Interaction 整头屏蔽；grounding 三档落库标记 |
| 12 | §9.2 Input-Binding Diagnostics（donor 匹配采样 + 四指标） | **not implemented** | corruption 生成器（family×variant 内 donor 匹配）+ eval 独立通道；严禁混入 matched-action-unit / do() 数据结构 |
| 13 | `execution_regime` 属性 + Q6–Q9 core 路由 + Q6 二分类 + Q9 六门（v1.3.3） | **not implemented**：`_realized_state_eligibility` 仍放行碰撞（benchmark.py:686）、Q9 门无碰撞前提（:1013）、Q6 三选项含死选项（:1108） | compiler / validator / eval 三侧改造；validator 从私有 rollout 重算 regime |
| 14 | §6.10 Post-contact Propagation track | **not implemented** | 联合向量 builder + joint/分阶段 scorer + `post_contact_diagnostic` 角色 |
| 15 | §12.4 发布硬门（label-support 分层、shortcut gate 接入编译、raw→published 损耗、candidate/formal 分离） | **partial**：`RELEASE_MIN_LABEL_SUPPORT` 存在但未按 regime 分层；`shortcut_audit` 为独立脚本未接编译门 | formal artifact 须经全部七门；preview 定级 candidate |

当前 preview `data/benchmark/egoconseq_v131a_preview_20260728` 已通过当前
artifact validator（10,536 items，0 violations）和 GT-as-pred 四头评分；它只验证
schema/编译/展示闭环，不满足 sensor pair、Q4 Direct/Propagation 和定向必要性样本的
正式发布配额。**v1.3.3 定级：candidate**——配对审计证实 Q9 99.4%/100% 落碰撞
rollout 且标签坍缩、Q6 closed 全 closer、Q1 76/24 失衡；其 `core` 标记无效，
任何数字不得外引（修订 10）。**现存数据不可只重编译**：按路由过滤后 Q9 forward_safe
仅剩 1 个安全样本、turn_choice 归零、Q6 的 90 个安全样本全 closer——必须按 §13.9
第 8–12 项定向补采。

### 12.3 已裁决的前开放点

1. do(body) 突出方式：不加第三个 headline 标量；Direct Body Effect 为 required primary
   intervention result，进摘要/主结果表/claim matrix。
2. 漏斗顺序：先全部连续 record，后定向 pilot，同一脚本复跑。
3. Q0a：保留公开 diagnostic track；评测 stateless。

### 12.4 发布硬门（v1.3.3；formal artifact 必须全部通过）

1. **Execution routing gate**：Q6/Q7/Q8/Q9 `score_role=core` 逐题独立验证
   `execution_regime == completed_clear`；
2. **Label support gate**：backend × family × variant 内检查标签分布与独立 scene
   支持；单标签层直接 fail（preview 的 Q6 全 closer 即触发此门）；
3. **Shortcut gate 接入编译**：§9.1 审计不再只是独立脚本——majority / action-only /
   Q1-proxy 攻击超门即阻断 formal artifact 产出；
4. **Publication role gate**：`contact_truncated` 的下游题必须标
   `post_contact_diagnostic`，validator 拒绝其 core 标记；
5. **Raw→published 损耗报告**：每 family×variant 报告 raw → completed-clear →
   evidence-qualified → balanced 的逐级损耗（防过滤器静默改变构念；§9.3 配套）；
6. **Validator 重算**：`execution_regime` 与 core 资格从私有 rollout 独立重算，
   不信任 item 声明字段；
7. **Candidate / formal 分离**：未过全部门的 artifact 只能标 candidate（当前
   preview 即是）；formal 标记须由通过记录背书。

---

## 13. 八项设计裁决（v1.2，构念定稿）

### 13.0 裁决依据：首次统一漏斗审计（2026-07-27）

`data/audits/design_funnel_v1.json`（工具 `pipeline/funnel_audit.py` +
`scripts/audit_design_funnel.py`，实现文件当次未提交，provenance 标
`usable_for_candidate_freeze: false`，须提交后干净重跑）。要点：

| 量 | 值 |
|---|---|
| 输入 → 去重 | 456 records → **52** Q4 physical groups / **12** scenes（23 个 v6 shard 接受，73 个旧 v5 拒绝） |
| 四半径 rollout 完整 | 32 组 |
| **四半径 direct flip** | **0**（20 个 flip 全部来自旧三半径协议）；rank 支持 0/0/0 |
| Propagation | flip-based 40 pairs / 20 groups / 7 scenes；both-collide 26 pairs / 21 groups / 7 scenes（Δendpoint ≥0.05 m：21/17/6） |
| both-collide 分类 | **全部 unresolved**——旧记录缺每半径物理接触 instance attribution → BC-simple = BC-branching = 0 |
| Sensor | Q4 multi-profile groups = 0；普通 outcome 240 组：Interaction 240/240 全充分；Endpoint & Target Relation 30 全充分/112 混合/98 全不足；Future Observation 12/48/180；Local Affordance 70/106/64 |
| Q6 geodesic 原始可用率 | GS 0/2378；HM3D 3579/4004 ≈ 89.4%；R2R 2901/2986 ≈ 97.2% |

结论：现有数据**不能**发布新版 Q4、不能授予任何 Propagation Tier；定向采集是主路径。

### 13.1 身体范围 → **收窄主张为平面 footprint**

oracle 是 2D ground disc，半径是唯一身体变量。论文用语固定为 "planar body footprint
(disc radius) conditioning"，不宣称三维身体/高度/形状理解。body height / 3D cylinder
oracle = vNext 独立 feasibility pilot，在 oracle 支持竖直碰撞前不公开任何高度字段。

### 13.2 弃权 → **确认取消（用户决定，终局）**

sufficient-only forced choice 定稿；selective prediction / abstention /
safety-calibration 不进论文主张。Evidence GT 完整存储，仅作发布门与选择偏置审计基础。

### 13.3 Propagation GT → **半径无关的中心距离（见 §7.2 冻结定义）**

`d_i` = 圆盘中心@realized endpoint 到固定目标参考集（record 级存储、与半径无关的目标
地面支撑点集）的最近点 2D 距离。禁用 `final_surface_distance_m`。新采集必须持久化
目标点集与每半径接触 instance id；旧记录只作质心代理分析。

### 13.4 Q4 数值协议 → **程序冻结，数值待定向 pilot**

默认**单一固定四半径网格**（选项安全按构造成立）；网格与 δ_abs/ρ_rel 由定向四半径
pilot 定（现存 flip=0，必须新采）；rank 1/2/3 支持按 **backend × unique scene** 计，
三档各 ≥ 预注册下限；grid bank 仅当单网格 yield 低于预注册下限时启用，且 bank 的
rank 均衡必须过 option-only 实证门。

### 13.5 Propagation Tier 规则 → **规则冻结，判级待定向数据**

每 stratum（flip-based / BC-branching）内，large-closer 与 large-farther **各**需
≥ N_min 个独立 scene（N_min 预注册默认 5，定向 pilot 做 power 校验后终定）；
BC-simple 永为诊断；BC 分类以每半径物理接触 instance id 判定（同段同源 + 放大率
< A_min = BC-simple；否则 BC-branching；A_min 由 pilot 分布定）。当前数据判级
不可用（13.0）。

### 13.6 Sensor 配对 → **门冻结，定向补数**

Q5a required heads = Interaction / Endpoint & Target Relation / Local Affordance（§8.4）；每 head
发布 Sensor Physical Invariance 前须达预注册最低 both-sufficient pair scene 支持。
审计显示瓶颈在 Endpoint & Target Relation（30 全充分）与 Future Observation（12），且 Q4 组
multi-profile = 0 → 定向采集必须包含 Q4 sensor siblings 与 ETR/FO 头的配对补数；
profile 选择偏置审计（§9.3）随行。

### 13.7 Q6 geodesic → **Endpoint & Target Relation 的 closed-only variant，有条件计分**

非独立头。仅 HM3D/R2R（GS 发布即 release failure，审计证实 GS 可用率为零）。
状态=诊断，navmesh-vs-surface 冲突来源审计通过后升为计分 variant；冲突样本只进独立
Euclidean–Geodesic Divergence 诊断集。

### 13.8 Headline 与 body 叙事 → **单 headline + 必报主结果（终局）**

唯一 headline = Visible Consequence（四头）；Direct Body Effect 是 **required primary
intervention result**（摘要、主结果表、claim matrix 必报，非附表切片）；Body
Propagation 按 Tier 限定措辞；Integration 单列。定位语见 §0。

### 13.9 定向采集规格（由 13.0 缺口直接导出）

1. **四半径 flip**：新协议四半径 rollout + 真翻转，per-backend rank 支持达 13.4 下限；
2. **可分类 both-collide**：持久化每半径物理接触 instance id + 目标地面支撑点集，
   away + 大身体早停单设预算（large-closer 稀缺源）；
3. **Q4 sensor siblings**：Q4 组全 profile 渲染（当前为 0）；
4. **ETR / FO 配对补数**：对两 profile 皆可指认目标的 pose 定向补采；
5. **Q8 设计配对**：FOV 对（do(sensor) 型）与遮挡对（matched do(action) 型）分开配额（§6.3）；
6. **起点五探针**：补采起点位姿的五探针掩码，支撑 Local Affordance 的
   起点/终点掩码必要性过滤器（§2）；
7. **Q9 掩码对**：matched 动作对产生不同 endpoint 掩码的定向搜索
   （Local Affordance 的正式 action-conditioning 证据；被第 10 条收紧取代）；
8. **安全 toward / lateral / away 动作**（v1.3.3，Q6/Q7 主榜）：completed_clear 下
   closer 与 farther 在每 backend 有独立 scene 支持（Q6 二分类标签配额）；
9. **安全 Q9 掩码变化样本**（v1.3.3）：completed_clear 且 endpoint_mask ≠
   start_mask 的定向搜索（安全走入死角/窄区）；forward_safe / turn_choice 标签
   配额随行；
10. **双安全 matched 掩码对**（v1.3.3，取代第 7 条）：配对两端均 completed_clear、
    终点掩码不同；
11. **Q8 安全设计配对**（v1.3.3）：第 5 条的 FOV 对与遮挡对全部限两端
    completed_clear；
12. **Q1 标签配额**（v1.3.3）：safe/collision 池平衡（当前 183/57 偏碰撞）；
    仍禁每 pose 固定 3+3（P0-1 不回退）。
13. **Q8 FOV 双方向预算**：proposal 分别识别 narrow-only 的像素支撑跨阈与
    wide-only 的视锥进入信号，按方向交错排序；真实 sibling 渲染审计须继续到两个
    realized 方向都命中或 proposal budget 耗尽。三后端 pilot 若任一方向未达到
    `5 scenes × 2 backends`，该方向只作 diagnostic，不得用另一个方向的数量替代。
    2026-07-29 的 `be2c0c0` 三后端 20-scene pilot 实测 realized
    wide-only = HM3D 20 / R2R 17 / GS 8，narrow-only 三后端均为 0；
    因此 wide-only 进入 primary Q8 FOV slice，narrow-only 固定为
    diagnostic-only，直到未来独立 pilot 重新满足同一预注册门。
14. **Q8 occlusion 单位化漏斗**：matched action pair 的筛选链与
    `matched action pair × target` 的筛选链分别单调；target expansion 是显式换轨点，
    禁止把两种单位拼成一条累计拒绝曲线。`be2c0c0` 三后端 20-scene
    pilot 实测 realized occlusion witness = HM3D 3 / R2R 6 / GS 0；
    这只证明构型可达，不满足三后端 primary 支持，须由 post-fix supplement
    继续补数，并在报告中保留 GS 零产出 stratum。

**补采优先序（2026-07-28 candidate 审计后定，以"尽快看到完整初版"为目标）**：
① Q6/Q7/Q8 completed-clear 定向样本（先补齐两个空缺 headline 头）→ ② 第 9 条安全
终点窄区/死角样本（覆盖 safe/unsafe 与 left/right/neither/both）→ ③ Q2 阶段配额
（complete / forward-1 / forward-2 / forward-3，按 forward-leg count 分层）→
④ 第 12 条 Q1 全局配额 → ⑤ GS 独立 scene 配额（Q1–Q9 各族覆盖，不能只有 Q4）→
⑥ do(action) / do(sensor) 配对与 Q10。

执行顺序：review + 提交漏斗审计四文件 → 干净工作区重跑（provenance 转 usable）→
定向采集 → 同一脚本复跑 → 数值冻结，文档升 v2.0 正式 publication contract。

### 13.10 Formal source v2 信任边界（2026-07-30）

正式编译输入不再仅凭 `records.jsonl` 哈希入选。唯一可入选来源须在
`docs/runs.json` 中显式启用、状态为 `completed`，且每个 shard 都由同一
`egoconseq.pose-pool.v2` shared pool 生成。source manifest v3 同时绑定并在
编译入口重验：registry 自身钉定的 backend/code revision/pool 声明及每个
shard 的 records 摘要、`run_meta.json` 摘要、record count 和模型输入 RGB
集合摘要，以及 pose-pool
manifest 与其索引的 RGB/geometry 资产。consumer 只读取 repository root 下
固定的 `docs/runs.json`，manifest 无权选择另一份 registry；source 被撤权后旧
manifest 立即失效。artifact compiler 对 records 与 run metadata 各只读取一次，
并对同一份内存字节完成摘要、解码、验证和 provenance 提取，禁止两次打开之间
的 ABA 替换；公开输入图也只读取一次并从同一份认证字节物化。`docs/runs.json`
是人工维护的信任根，修改它本身即是授权变更，不属于 untrusted manifest
攻击。inline、interrupted、dirty、v1 pool、路径逃出仓库、未声明 shard
或任一摘要漂移均为 formal-blocking failure；不提供旧 manifest adapter。

### 13.11 Oracle v7 GT 修订（2026-07-30）

本节只修正 GT 实现与信任边界，不改变五个结果头、公开输入、动作语义或
publication margin。当前契约升为 `conseq.v9` /
`ground-disc-visible-v8` / `egoconseq.qa.v15` /
`egoconseq.pose-pool.v3`；题面未变，prompt contract 保持 v14。

四项 breaking 修订：GS navmesh snap/unsupported-floor 不再伪装成碰撞；
Q3 使用圆盘中心到真实接触表面的半径外推并独立检查方向；Q4 连续搜索 bracket
与公开四半径网格分离；Q8 将视锥内的零像素与非零但过小分别标为 occluded 与
too_small。Artifact validator 必须核对公开输入与 private rollout 一一对应，
并从 private source facts 重算 Q0a、Q1--Q10，而不是信任 stored answer。旧
records、pose pools 与 QA artifacts 一律拒绝，不设 adapter。完整验收见
`docs/2026-07-30-oracle-v8-acceptance.md`。

### 13.12 Oracle v8 GT 修订（2026-07-30）

本节只修正物理 oracle 的实现与离散化约定，不改变五个结果头、公开输入、动作语义或
publication margin。契约升为 `ground-disc-visible-v8` /
`radius_extrapolated_navmesh_boundary_v2`；`conseq.v9`、`egoconseq.qa.v15`、
`egoconseq.pose-pool.v3` 不变。v7 records、pose pools 与 QA artifacts 一律拒绝，不设 adapter。

四项修订：

**(1) HM3D/R2R 的无效几何不再冒充碰撞。** v7 只为 GS 实现了这条；navmesh 后端把任何
`is_navigable=False` 都当碰撞，于是超过 `NAV_Y_DELTA_M` 的楼梯落差会发布成 Q1 碰撞加 Q2
停止距离。`Nav.query_pose` 现在用**同一个 float32 点**同时喂 `is_navigable` 与 `snap_point`，
并保守四分：投影非有限 → `unsupported_floor`；横向偏移 ≥ habitat 自身的 1 cm 容差 →
`None`（真实横向碰撞，必须保持碰撞）；两个子句都解释不了 → `navmesh_boundary`。
navmesh 按半径重建，所以"圆心仍能以 ≤1 cm 投影到多边形上"蕴含该半径下不可能与障碍重叠。
`validate.py` 的 collision-source 分类同时对两个 authority 生效，navmesh 碰撞不携带正向 source。

**(2) 对齐三个 oracle 的离散化约定。** Recast 先体素化再腐蚀，沿用 habitat 默认会把
0.30 m 机体量化成 `ceil(0.30/0.2)=2` 格 = 0.40 m，且 0.2 m 的可跨越高度让 navmesh 跨过
depth 与 Gaussian 都判为实心的障碍。`NAVMESH_CELL_SIZE_M` / `NAVMESH_CELL_HEIGHT_M` /
`NAVMESH_MAX_CLIMB_M` 全部显式设为 0.05 并进入 navmesh 缓存 key 与文件名，使
`GROUND_OBSTACLE_BAND_M` 两端都落在整格上；depth 与 GS 的下界统一改为半开。
**这是对齐离散化约定，不是让三者成为同一个身体**——三者仍是三套离散化。

**(3) Q4 的转折区间由四个公开半径的真实结果唯一决定。** Recast 按 `ceil(radius/cell_size)`
腐蚀，二分出的连续临界半径只能分辨到一个格，不能作为标签权威。rank 现在读自重跑出的
outcome vector（`S S C C → rank 2`），`critical_radius_m` 只用于 proposal 与边界稳定性筛选；
`SSSS` / `CCCC` 只出 open control，非单调或含 invalid geometry 一律拒绝。
公开四半径必须落在 0.05 m 格点上且两两占据不同格。
发布措辞为"四个固定身体半径等级之间的安全→碰撞转折区间"；GS 走连续 Gaussian geometry，
不适用量化说法。

**(4) Q3 接触面 v2。** 点可以越过腐蚀多边形约 1 cm 仍读作 navigable，所以二分收敛到的是
这条容差等值线而非多边形边缘；接触中心到配置边界的距离约为一个容差加二分残差。
`CONTACT_NAVMESH_BOUNDARY_MAX_M` 改为由
`NAVMESH_LATERAL_SNAP_MAX_M + MARCH_STEP_M / 2**CONTACT_REFINE_ITERS +
CONTACT_SURFACE_RADIUS_TOL_M` 派生。外推距离从整个半径改为 `radius − norm`
（圆心已越界 norm，圆盘在等值线处接触障碍），`distance_m` 同步，并加 `0 < norm < radius` 守卫。
validator 用同一公式重算——这是**重算构造式以检测篡改**，不是独立几何验证。
