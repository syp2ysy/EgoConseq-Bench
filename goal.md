我理解你们真正想做的东西了：**不是让模型帮机器人规划路线，也不是问“这个门能不能走”这种语义 affordance，而是问：给定当前第一视角图、机器人的身体 footprint、一个短动作，模型能不能在脑子里把身体扫过去，判断局部物理后果。**

这个 benchmark 是合理的，而且比“泛空间智能”更有机会。但设计必须非常克制：**所有问题都要能严格还原到一个几何算子**，否则很容易又变成 R2R action success、导航、楼梯理解、房间物体识别。

核心定义我建议钉死成：

```text
d_safe(profile, action)
= 从当前 pose 出发，agent body 按 action 形成 swept volume，
  与当前可见/可直接验证的场景几何第一次接触前的距离。
```

v1 先简化：

```text
profile = circular footprint radius r, fixed height
action = forward / turn-left θ then forward / turn-right θ then forward
GT = per-radius navmesh + swept first-contact oracle
```

模型输入永远只有：

```text
egocentric RGB image + question
```

depth、pose、navmesh、semantic、top-down、trace 都只能用于离线 GT 和展示。

---

# 1. 先看别人 benchmark 的分类逻辑：你们应该学什么

相关 benchmark 的共同经验是：**分类不能按物体/房间/场景元素分，而要按能力轴分。**

PhysBench 把 physical world understanding 分成 physical object properties、object relationships、scene understanding、physics-based dynamics 四大域，并继续细成多个 subclass / capability dimension。它的价值不是“题多”，而是每个域对应一个清楚的物理能力缺口。([arXiv][1])

3DSRBench 也不是简单列“左/右/前/后”题，而是围绕 3D spatial reasoning 的 height、orientation、location、multi-object reasoning 等能力设计，并强调 answer balance 和 FlipEval / complementary pairs 来防止语言先验捷径。([arXiv][2])

Spatial457 更像你们应该参考的“诊断型 benchmark”范式：它设置多种 question types 和 difficulty levels，从简单识别逐步到 3D orientation、6D spatial reasoning 和 collision prediction，用难度阶梯暴露模型在哪个空间能力层级崩。([arXiv][3])

VSI-Bench 的启发是：模型输入可以只给视觉观察，但离线可以用 RGB-D、mesh、point cloud 或 pose 生成高质量空间 QA；关键是 QA 要 visually grounded，并且要过滤语言/常识捷径。VSI-Bench 本身是 egocentric video spatial intelligence，不是你们的单图 ego-body action consequence，但它的“离线几何生成、前台只给视觉”的范式值得学。([Emergent Mind][4])

和你们最近的两个工作是 TouchSafeBench 与 CapNav。TouchSafeBench 测的是 collision grounding，但它是 Habitat 3.0、多视角 RGB-D、episode/trajectory/contact label，核心是 present/imminent contact safety；你们要隔离的是单张 RGB 下的 hypothetical ego action swept-volume consequence。([arXiv][5]) CapNav 是 capability-conditioned route-level navigation，输入 tour video、navigation graph、agent mobility profile，评估 task feasibility、path validity、route traversability 等；你们没有 route、goal、SR/SPL，只问当前图中一个短动作的局部身体扫掠后果。([arXiv][6])

所以你们的分类原则应该是：

```text
主分类 = MLLM 哪种 ego-action consequence 能力
分析轴 = body size / action type / geometry type / difficulty
不是按房间、物体、楼梯、门、走廊做顶层分类。
```

---

# 2. 你们最强的 paper highlight 应该是什么

我建议 headline 这样写：

> **EgoConseq-Bench tests whether MLLMs can mentally sweep their own body through the visible egocentric space under a candidate action.**

中文：

> **EgoConseq-Bench 测的是：MLLM 能不能从一张第一视角图里，把自己的身体 footprint 和候选动作投进当前可见空间，预测身体扫过去之后会不会接触、还能走多远、哪边更安全。**

真正的 highlights 可以是四个：

1. **From seeing space to sweeping body through space**
   不是识别障碍，也不是说“前面有门”，而是判断“我这个身体沿这个动作扫过去会发生什么”。

2. **Body as a causal collision operator**
   body width 不是 prompt 里的装饰参数，而是会改变 GT 的因果变量。小 body 能过、大 body 不能过，是你们最强证据。

3. **Single-image, visible-local, no hidden-space guessing**
   只问当前 RGB 可见或直接可验证的局部空间，不让模型猜转角后、遮挡后、画面外。

4. **Deterministic geometry oracle, not LLM labeling**
   标签来自 Habitat navmesh / swept-volume first-contact，LLM 只参与问法设计，不许幻想 GT。

这四条比“我们做了一个安全 benchmark”强很多。

---

# 3. Benchmark 分类：我建议 5 个核心能力 + 1 个可选诊断

不要把问题分成“门、走廊、楼梯、椅子”。这些只是 geometry tag。主分类应该是下面这些能力。

## C1. Forward Swept Clearance：向前扫掠距离估计

**测什么：**
模型能不能判断这个身体如果一直正向前移动，第一次身体接触前大概能走多远。

**GT：**

```text
label = d_safe(r, forward)
```

**答案形式不要只做选择题。可以有两种版本：**

结构化数值版：

```text
请回答一个数字：大约可以前进多少个机器人身宽？
```

离散版：

```text
A. 少于 2 个身宽
B. 2 到 4 个身宽
C. 4 到 8 个身宽
D. 多于 8 个身宽
```

**低分说明：**

```text
模型可能能看见“前方有空间”，但估不出这个空间相对自己身体到底够不够。
```

注意：不要问“机器人计划走 0.25m，实际能走多少”。那会退化成 rollout completion 或读题干数字。

---

## C2. Fixed-Horizon Contact Prediction：固定动作距离内会不会接触

**测什么：**
给定一个短动作，模型判断身体 sweep 在固定 horizon 内是否会发生物理接触。

**动作可以是：**

```text
forward H
turn-left 15° then forward H
turn-right 15° then forward H
turn-left 30° then forward H
turn-right 30° then forward H
```

**GT：**

```text
contact = d_safe(r, action) < H
```

**问题示例：**

```text
机器人是圆柱形底盘，宽约 0.5 米。
只看这张第一视角图。如果它先向左转 30°，再向前移动 2 个身宽，
在当前可见局部空间内，身体会不会发生物理接触？
请回答：会接触 / 不会接触。
```

**低分说明：**

```text
模型不是完全看不懂图，而是不能把动作轨迹投进图像空间做 swept-volume 判断。
```

这里要非常严格：如果 Habitat rollout “提前停止”，但没有可见 contact evidence，这个样本不能作为 contact 题。你们测的是 physical contact，不是 action executor 是否完成。

---

## C3. Directional Swept-Volume Comparison：多方向动作后果比较

**测什么：**
模型能不能比较左前、正前、右前哪个方向的 swept volume 更安全。

**GT：**

```text
argmax_theta d_safe(r, theta)
theta ∈ {-30°, 0°, +30°}
```

**问题形式可以是 ranking，不一定选择题：**

```text
机器人宽约 0.5 米。
比较三个方向：左前 30°、正前方、右前 30°。
请按“第一次接触前能移动的距离”从长到短排序。
```

模型输出：

```text
右前 > 正前 > 左前
```

评分可以用：

```text
top-1 accuracy
Kendall tau / pairwise ranking accuracy
```

**低分说明：**

```text
模型不会比较多个候选动作的物理后果，可能只是有 center bias，总偏向正前方。
```

要注意：如果 top1 和 top2 的 `d_safe` 差距太小，比如小于 0.5 body-width，这题应该丢弃，除非显式允许 “差不多”。

---

## C4. Footprint Counterfactual Sensitivity：同图换身体宽度，答案是否翻转

这是你们最强的一类。

**测什么：**
同一张 RGB、同一个 pose、同一个动作，只改圆柱形底盘的 body radius / width，GT 是否翻转；模型是否也翻转。

注意题面形式：**每道题只描述一个机器人、一个底盘直径**。反事实不是在同一道题里做双对象比较，而是在 manifest 里把同图同动作、不同底盘直径的若干道题用同一个 `group_id` 绑起来，组内看答案是否翻转。机器人高度固定为 Habitat 默认 agent height，不作为变量，也不写进 prompt。

**GT：**

```text
label(r) = contact iff d_safe(r, action) < H
group flip = 同一 group 内同时出现 contact 和 no_contact
```

**问题示例：**

```text
图中是一个圆柱形底盘的机器人，底盘直径约 0.8 米。
它从当前位置朝正前方移动约 2.0 米。
只看这张第一视角图，判断它的身体在当前可见的局部空间内会不会与障碍物发生接触。
请只回答：会接触 / 不会接触。
```

同一个 `group_id` 下可以另有一道完全相同图像/动作、但底盘直径约 0.2 米的题。窄底盘 no_contact、宽底盘 contact 的组才是 O5 的核心判别组。纯 footprint radius 下 `d_safe(r_large) <= d_safe(r_small)`，这个单调性仍然作为 oracle sanity check。

**低分说明：**

```text
模型没有把 body footprint 当成 collision operator，而是把 body 参数当成无关文字。
```

这是最像 paper finding 的地方：**同图只改 body width，GT 翻转，模型不翻转。**

---

## C5. Critical Body Width / Passability Limit：最大能通过的身体宽度

这个是我建议你们新增的强任务，因为它不是选择题，而且非常贴合“机器人身体参数影响动作后果”。

**测什么：**
模型能不能估计当前可见通道最多允许多宽的圆柱身体通过。

**GT：**

```text
r_crit = max r such that d_safe(r, action) >= H
critical_width = 2 * r_crit
```

用 Habitat 可以通过对 radius 二分搜索 navmesh 或 swept-corridor passability 得到。

**问题示例：**

```text
只看这张第一视角图。
如果机器人沿正前方移动 2 米，估计圆柱底盘最大宽度约是多少，
才能在当前可见局部空间内不发生身体接触？
请回答一个米数，例如 0.4 米。
```

或者更稳一点，用相对单位：

```text
请回答最大宽度大约是当前机器人宽度的多少倍。
```

**低分说明：**

```text
模型不是不会选 small/large，而是缺少连续的 body-clearance sensitivity。
```

这类任务能让你们摆脱“全是选择题”的问题，也能更有 benchmark 味道。

---

## C6. First-Contact Grounding：首碰区域定位，可选诊断

**测什么：**
如果动作会接触，模型能不能指出首碰大概发生在图像或 swept corridor 的哪一侧。

**GT：**

```text
collision_point_3d → project to RGB → region / grid cell
```

**问题形式不要只做 lower-left / center / right 三选一，可以做 grid answer：**

```text
把图像分成 3×3 网格，行从上到下为 1/2/3，列从左到右为 A/B/C。
如果机器人执行该动作，第一次身体接触最可能出现在第几个网格？
请用 A1 到 C3 回答；如果查询范围内没有接触，回答 none。
```

**低分说明：**

```text
模型即使能猜“会撞/不会撞”，也没有把答案 grounded 到可见首碰证据。
```

这类我建议作为诊断，不作为 v1 生死门核心，因为它的标注和可见性过滤更容易 noisy。

---

# 4. 任务形式不能全是选择题：建议 mixed structured QA

你们说“不能都是选择题”非常对。但完全 free-form 也不现实，因为自动评分会乱。最佳方案是 **structured but non-MCQ**。

建议每类至少有一种非选择题形式：

| 类别                       | 推荐答案形式                         | 评分                             |
| ------------------------ | ------------------------------ | ------------------------------ |
| Forward Swept Clearance  | 数值：`3.2 body-widths`           | MAE / bin accuracy             |
| Fixed-Horizon Contact    | `contact / no contact`         | Acc / False-Safe Rate          |
| Directional Comparison   | 排序：`left > straight > right`   | Top-1 / Kendall tau            |
| Footprint Counterfactual | 单题：`contact / no_contact`；组内：flip / invariant | case Acc / group flip consistency |
| Critical Body Width      | 数值：`0.55m` 或 `1.2× body-width` | MAE / tolerance accuracy       |
| First-Contact Grounding  | grid cell：`B3` 或 `none`        | grid accuracy / pixel distance |

这样既不像生硬选择题，又能自动评估。

你们可以在 HTML 展示页给自然问法，但 manifest 里要保存标准化 answer schema，例如：

```json
{
  "answer_type": "numeric_body_widths",
  "unit": "body_width",
  "label": 3.6,
  "tolerance": 0.75
}
```

或者：

```json
{
  "answer_type": "ranking",
  "items": ["left30", "straight", "right30"],
  "label": ["right30", "straight", "left30"]
}
```

---

# 5. Action library：必须物理自洽，不能像导航指令

你们的动作应该是短动作 primitive，而不是路线。推荐：

```text
A0: forward H
A1: turn-left 15° then forward H
A2: turn-right 15° then forward H
A3: turn-left 30° then forward H
A4: turn-right 30° then forward H
```

H 用 body-width 做主单位：

```text
H ∈ {1, 2, 4, 6} body-widths
```

米制可以在 prompt 中出现，但不要作为唯一尺度。因为单张 RGB 下绝对米制会把问题混成 monocular metric depth estimation。更稳的是：

```text
机器人宽约 0.5 米。如果它向前移动 4 个身宽，也就是约 2 米……
```

但注意不要在 Forward Clearance 题里给“计划距离”，否则会泄答案。

对你给的例子：

> 机器人在当前往前走 0.5m，再左转30度，再走0.25m会不会发生碰撞？

这是合理的，但它属于 **compound swept trajectory**。GT 不能由 rollout success 给出，而要把路径拆成 piecewise swept volume：

```text
segment 1: forward 0.5m
turn: heading changes by 30°, circular body 原地转动 footprint 不变
segment 2: forward 0.25m along new heading
first contact = min contact over both forward segments
```

并且必须满足：

```text
整个 swept path 仍在当前 RGB 的可见/可验证区域内
```

否则就变成让模型猜转头后看到什么。

---

# 6. Geometry / environment 分类：只能做 tag，不做顶层类别

环境应该按 **局部几何交互类型** 打 tag，用于结果分桶，不要作为主分类。

建议 geometry tags：

| Geometry tag              | 作用                | 可用于哪些任务  |
| ------------------------- | ----------------- | -------- |
| `open-free-space`         | 控制模型是否过度保守        | C1/C2    |
| `frontal-barrier`         | 前方墙、柜、障碍          | C1/C2/C6 |
| `narrow-gap`              | 宽度临界，适合 body flip | C2/C4/C5 |
| `doorway-threshold`       | 真实机器人常见           | C2/C4/C5 |
| `asymmetric-side-opening` | 左/右/直差异明显         | C3       |
| `corner-branch`           | 方向选择              | C3       |
| `clutter-near-floor`      | 椅腿、箱子、家具边缘        | C1/C2/C6 |
| `dead-end`                | 前方短距离必碰           | C1/C2    |

第一版不要把 stairs、drop-off、rail、overhang 当核心，除非你们扩展 locomotion 和 3D body envelope。你提到“正对面是楼梯扶手区域，让它往前多少米都是正常的”——这种样本必须非常小心：

* 如果是**地面平面上的栏杆/扶手支柱阻挡 footprint**，并且 navmesh / mesh first-contact 能检测到，可以收。
* 如果涉及**楼梯能不能爬、会不会跌落、轮子能不能上台阶、扶手高度碰不碰身体上半部分**，v1 应该丢弃。
* 如果要做这些，必须定义 locomotion capability 和 vertical body envelope，那已经是 v2，不是 v1 的 ground-plane swept footprint。

---

# 7. Habitat pipeline：应该这样设计

Habitat-Sim 很适合做这件事，因为它支持可配置 agent、RGB-D 渲染、3D 场景和 navmesh/pathfinder；官方文档里 NavMeshSettings 的 `agentRadius` 就是用于把 walkable area 从障碍物处 erosion / shrink，PathFinder 则负责 navmesh sampling、path finding、collision、island query 等。([AI Habitat][7])

## Step 1：场景和 agent profile

v1 只做圆柱 footprint：

```text
height fixed = 1.5m
camera height fixed = 1.5m
radius ∈ {0.10, 0.25, 0.40}
width = 2r
```

每个 radius 都要重新生成 navmesh：

```text
NavMeshSettings.agentRadius = r
NavMeshSettings.agentHeight = fixed_height
```

不要拿默认 navmesh 再额外扣 clearance，否则会 double-count radius。

## Step 2：采样 pose/yaw

不要随机采完就用，要分层采样：

```text
scene quota
floor / island quota
geometry tag quota
answer label quota
body radius quota
action angle quota
difficulty quota
```

每个起点必须满足：

```text
start pose navigable for the relevant radius
RGB 有足够地面/前方区域
depth 有效
不是贴墙极近
不是纯大空地
不是强遮挡/画外决定
```

对于 counterfactual：

```text
start pose 必须对 small / medium / large 都合法
```

## Step 3：计算 d_safe

对每个 `(pose, yaw, radius, action)`：

```text
1. 根据 action 得到 swept path:
   forward: heading = yaw
   turn-left/right then forward: heading = yaw ± θ
   compound: 多段 heading 拼接

2. 沿 path 以 0.02m 或 0.05m step march center point

3. 对 radius-specific navmesh:
   if pathfinder.is_navigable(center_point) == false:
       first_contact = 当前距离
       stop

4. 若到 D_max 都没 contact:
       d_safe > D_max
```

这个是 v1 的 ground-plane swept-cylinder oracle。它不是完整 3D physics，但足够支撑你们第一版“身体 footprint + visible local action consequence”。

如果要做桌下、横梁、机器人身高、扶手高度，必须进入 v2：

```text
3D mesh / depth point cloud
+ vertical cylinder / box body
+ full swept-volume intersection
+ profile-specific camera height re-render
```

## Step 4：可见性过滤

每个样本必须过可见性门：

```text
visible_sweep_ratio >= 0.7
first_contact_visible = true       # contact case
requires_hidden_geometry = false
FOV_supported = true
margin >= 0.5 body-width
```

具体做法：

```text
将 swept corridor footprint 采样点投影到当前 RGB
统计 corridor 中可见比例
将 first_contact point 投影到像素
用 depth buffer 检查投影深度是否一致
如果 contact / no-contact 的判断依赖画外、遮挡后、转角后，丢弃
```

no-contact 题也要谨慎：不能因为 navmesh 上 6m 可通就问模型“能不能走 6m”，除非这 6m 的 corridor 在当前图中可见或直接可验证。

## Step 5：任务实例化

从同一个 oracle 输出派生所有任务：

```text
Forward Clearance:
  label = d_safe(r, 0)

Fixed-Horizon Contact:
  label = d_safe(r, action) < H

Directional Ranking:
  label = sort_by d_safe(r, -30), d_safe(r, 0), d_safe(r, +30)

Footprint Counterfactual:
  label = compare pass(r_small), pass(r_large)

Critical Body Width:
  label = binary_search max r satisfying d_safe(r, action) >= H

First-Contact Grounding:
  label = project(first_contact_3d) to grid / corridor side
```

这样所有分类都挂回同一个核心算子，不会散。

## Step 6：dataset-level balance 和去重

必须有 dataset 级 gate：

```text
每类单一答案占比不能过高
每个 scene 最多占总样本一定比例
同一 RGB 不能跨不同 ability category 重复出现
同一 trajectory 最多贡献 1 个非 counterfactual case
counterfactual group 允许同图复用，但必须标 group_id
```

建议配额：

```text
每个核心类别至少 100 cases 才能跑正式 baseline
demo 每类至少 30 cases
counterfactual 至少 50 pairs
```

HTML showcase 可以每类展示 10 个，但不能把 10 个当 benchmark。

## Step 7：质量检查

必须跑这些 sanity check：

```text
radius monotonicity:
  r_small < r_large => d_safe(small) >= d_safe(large)

step-size stability:
  step=0.05m vs 0.025m label 一致率 >= 95%

horizon margin:
  |d_safe - H| >= 0.5 body-width

ranking margin:
  top1 - top2 >= 0.5 body-width, unless label is "tie"

blind baseline:
  text-only 接近随机

majority / radius-only / action-only:
  明显低于 RGB 模型或接近随机
```

如果这些不过，说明不是模型差，是数据设计有问题。

---

# 8. Prompt 设计：要像机器人真的会问的问题

你们现在最需要避免的是“simulator log 翻译成人话”。问题要像一个机器人在行动前问自己的安全预测。

## 好 prompt 的原则

使用：

```text
physical contact
body width
visible local space
from the current view
```

避免：

```text
clear
blocked
safe
dangerous
open corridor
solid obstacle
walkable
success
rollout
progress ratio
```

因为这些词要么泄答案，要么把问题带回导航/affordance。

## 示例 1：Forward 数值题

```text
机器人是圆柱形底盘，宽约 0.5 米。
只看这张第一视角图。如果它从当前位置一直正向前移动，
在第一次身体接触前大约能走多少个身宽？
请只回答一个数字。
```

标准答案：

```text
3.4
```

评分：

```text
absolute error <= 0.75 body-width 记为正确
```

## 示例 2：固定动作 contact

```text
机器人是圆柱形底盘，宽约 0.5 米。
只看这张第一视角图。如果它向右转 30°，再向前移动 2 个身宽，
在当前可见局部空间内，身体会不会发生物理接触？
请回答：会接触 / 不会接触。
```

## 示例 3：方向排序

```text
机器人是圆柱形底盘，宽约 0.5 米。
比较三个动作：左前 30°、正前方、右前 30°。
请按“第一次身体接触前可移动距离”从长到短排序。
```

## 示例 4：body counterfactual

```text
图中是一个圆柱形底盘的机器人，底盘直径约 0.8 米。
它从当前位置朝正前方移动约 2.0 米。
只看这张第一视角图，判断它的身体在当前可见的局部空间内会不会与障碍物发生接触。
请只回答：会接触 / 不会接触。
```

## 示例 5：critical body width

```text
只看这张第一视角图。
如果机器人要沿正前方移动 2 米，估计圆柱底盘最大宽度约是多少，
才能在当前可见局部空间内不发生身体接触？
请回答一个米数。
```

## 示例 6：first-contact grid

```text
把图像分成 3×3 网格，列为 A/B/C，行为 1/2/3。
如果机器人正向前移动到第一次身体接触，首碰位置最可能在哪个网格？
如果查询范围内没有可见接触，回答 none。
```

这些问题比“直走 5 步实际推进到什么程度”自然得多，也更贴近你们的能力目标。

---

# 9. Baseline 和指标必须按类别报告

不要只报 overall accuracy。你们要让每类低分都能授权一个结论。

| 类别                       | 指标                                  | 主要结论                        |
| ------------------------ | ----------------------------------- | --------------------------- |
| Forward Clearance        | MAE / bin Acc                       | body-relative clearance 是否弱 |
| Fixed-Horizon Contact    | Acc / False-Safe Rate               | 会不会把危险动作说成不接触               |
| Directional Ranking      | Top-1 / Kendall tau                 | 是否能比较多方向 action consequence |
| Footprint Counterfactual | group flip accuracy / invariance error | 是否对 body width 敏感           |
| Critical Width           | MAE / tolerance Acc                 | 是否理解连续 body-width 约束        |
| Contact Grounding        | grid Acc / normalized distance      | 是否 grounded 到可见首碰证据         |

最重要的指标我建议是：

```text
False-Safe Rate
= GT 会接触，但模型说不会接触的比例
```

因为你们的现实含义是 embodied brain 的行动前安全判断。

必跑 baseline：

```text
random
majority
blind text-only
radius-only
action-only
center-ray depth heuristic
floor/corridor-width heuristic
geometry oracle
MLLM RGB+question
MLLM RGB+sweep overlay
human small set
```

其中 MLLM + sweep overlay 是诊断：如果 overlay 后涨很多，说明模型不是不会读问题，而是不知道身体 footprint 会扫过哪里。

---

# 10. 你们应该明确不做什么

为了避免跑偏，我建议在 goal 里加一个 hard exclusion list：

```text
不使用 rollout completion 当 GT
不问 route success / goal reaching
不问“这条路能否到达目标”
不把楼梯/台阶/斜坡作为 v1 核心
不把 overhang / 桌下 / 高度碰撞放进 v1
不问遮挡后、转角后、画外区域
不让 prompt 出现会泄答案的场景描述
不让 LLM 生成 image-backed label
```

尤其要强调：

```text
rollout trace 可以用于 debug，但不能定义 label。
```

否则很容易又变成 R2R action success benchmark。

---

# 11. 推荐的最终 v1 方案

我会把 v1 定成这个样子：

## Core setting

```text
Simulator: Habitat + HM3D / MP3D
Input to model: single RGB + question
Offline signals: pose, depth, navmesh, top-down, trace
Body: circular cylinder footprint
Radii: 0.10 / 0.25 / 0.40 m
Height: fixed
Actions: forward, turn ±15/±30 then forward
GT: radius-conditioned swept first-contact distance
Scope: visible local space only
```

## Categories

```text
C1 Forward Swept Clearance
C2 Fixed-Horizon Contact Prediction
C3 Directional Swept-Volume Comparison
C4 Footprint Counterfactual Sensitivity
C5 Critical Body Width / Passability Limit
C6 First-Contact Grounding optional
```

## Analysis tags

```text
geometry_tag:
  open-free-space
  frontal-barrier
  narrow-gap
  doorway-threshold
  asymmetric-side-opening
  corner-branch
  clutter-near-floor
  dead-end

difficulty:
  easy / medium / hard based on margin

body_radius:
  small / medium / large

action_type:
  forward / left15 / right15 / left30 / right30 / compound
```

## DoD for demo

```text
每个核心类 >= 30 accepted cases
counterfactual >= 30 pairs
至少 5 个 scenes
同一 RGB 不跨 category 重复
visible_sweep_ratio >= 0.7
step-size label stability >= 95%
blind / majority 不高
human > MLLM
MLLM 与 oracle/human 有明显 gap
```

---

# 12. v2 可以做什么，但不要现在混进来

v2 再考虑：

```text
不同 camera height / body height
low robot vs service robot vs humanoid
桌下 / 横梁 / 悬空柜 / 门框高度
full 3D vertical body envelope collision
profile-specific RGB re-render
```

这时核心函数扩展成：

```text
d_safe(agent_profile, action)
```

其中 `agent_profile` 包含：

```text
body_radius
body_height
camera_height
camera_pitch
locomotion capability
```

但这不是 v1。v1 如果强行问身高，就会出问题，因为身高变了，egocentric image 也应该变。

---

# 13. 最终判断

你们应该这样做这个 benchmark：

```text
用 Habitat 生成单张第一视角图；
用 per-radius navmesh / swept first-contact oracle 生成 d_safe；
把所有 QA 都从 d_safe 派生；
用可见性过滤保证题目只问当前图能支持的局部物理后果；
用 counterfactual 和 answer balance 防捷径；
用 mixed structured QA 避免全是选择题；
按能力类别报告结果，而不是只报总分。
```

最重要的是别忘了这句：

> **EgoConseq-Bench 不是问模型“能不能走过去”，而是问模型“这个身体按这个动作扫过当前可见空间时，物理上会发生什么”。**

只要你们始终守住这个定义，这个 benchmark 是有意义的，而且能和 PhysBench、Spatial457、3DSRBench、VSI-Bench、TouchSafeBench、CapNav 拉开距离。

[1]: https://arxiv.org/abs/2501.16411?utm_source=chatgpt.com "PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding"
[2]: https://arxiv.org/abs/2412.07825?utm_source=chatgpt.com "3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark"
[3]: https://arxiv.org/html/2502.08636v4?utm_source=chatgpt.com "A Diagnostic Benchmark for 6D Spatial Reasoning of ..."
[4]: https://www.emergentmind.com/topics/visual-spatial-intelligence-benchmark-vsi-bench?utm_source=chatgpt.com "VSI-Bench: 3D Spatial Reasoning Benchmark"
[5]: https://arxiv.org/html/2605.31196v1?utm_source=chatgpt.com "Probing Collision Grounding in Vision-Language Models ..."
[6]: https://arxiv.org/pdf/2602.18424?utm_source=chatgpt.com "CapNav"
[7]: https://aihabitat.org/docs/habitat-sim/habitat_sim.nav.NavMeshSettings.html?utm_source=chatgpt.com "habitat_sim.nav.NavMeshSettings | Habitat Sim Docs"
