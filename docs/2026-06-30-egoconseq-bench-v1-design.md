# EgoConseq-Bench v1 设计文档（v1.0，冻结）

> 日期：2026-06-30
> 状态：**v1.0 冻结**。所有 OPEN 项已闭合，本文件为后续生成器/oracle/评测的**唯一事实源**；下游脚本不得自行写死 `CATEGORIES`、可见性阈值、GT 构造法、相机/身体参数等常量，必须从本文件派生。
> 收敛过程：多轮设计讨论 + 一轮我的 critique + 一轮外部 review（详见末尾「决策记录」）。
>
> **OPEN-1 已决**：v1 = 固定高度(1.5m) visible cylinder + 丢弃歧义样本；height/相机固定为 **Habitat-Sim 默认 agent(1.5m/0.1m/sensor 1.5m)**，不借任何机器人身份、只动 footprint radius；HFOV 79°（已定）；prompt 只强调 width 不提 height；§12 = "收可见稳定垂直障碍、丢弃 footprint-vs-cylinder 歧义"。

---

## 0. 一句话定义

**EgoConseq-Bench 测的是：MLLM 能不能从一张第一视角 RGB 里，把自己的身体 footprint 沿一个候选短动作投进当前可见空间，预测身体扫过去会不会接触、还能走多远、哪个方向更安全、换个身体宽度结论是否翻转。**

- **不是**导航：没有 path / goal / SR / SPL。
- **不是** PhysBench 式 object 物理属性。
- **不是** VSI 式被动空间关系。
- 是：**single-image + ego-body swept-volume + 仅可见局部 + body 作为因果变量**。

诚实边界：swept-volume = 标准圆柱碰撞检测，**本身不是 novelty**；novelty 在「探测 MLLM 能否从单帧做到这件事」+ counterfactual(body) 设计。

---

## 1. 核心算子（所有题的唯一来源）

```text
d_safe(r, action)
= 从当前 pose 出发，半径 r 的圆形 footprint 沿 action 的 piecewise swept path，
  在【当前帧 depth 反投影点云】中第一次接触前的距离（沿路径弧长）。
  仅取可见几何；超过 D_max 未接触记为 d_safe > D_max。
```

模型输入**恒为**：`单张 egocentric RGB + question`。
`depth / pose / navmesh / top-down / trace` **只能用于离线 GT 生成和可视化展示**，绝不进入模型输入。

---

## 2. 分类：操作轴 × 读出轴（两根正交轴）

> 设计决策：**不**把分类做成「门/走廊/楼梯/椅子」这种 geometry 顶层类，也**不**像旧 goal.md 那样把「几何操作」和「答案格式」混在一根轴上。
> 主分类 = **几何操作轴**（模型必须想象的空间变换）；答案格式降为**正交的读出 tag**（用于满足「不全是选择题」）。

### 2.1 操作轴（主分类）

| ID | 操作 | 考察的核心能力 | 主用读出格式 | 定位 |
|----|------|----------------|--------------|------|
| **O1** | 前向 clearance | ego-scaled 前向自由距离感知 | magnitude / threshold | 地基 |
| **O2** | 孔径 / 横向通过性 | 最宽可通过身体（侧向 margin） | magnitude | 地基 |
| **O3** | 转向后 sweep | 心理旋转 swept corridor 后判接触 | threshold | 中间 |
| **O4** | 跨朝向比较 | 比较多方向后果（相对、抗绝对尺度误差） | ranking | **headline** |
| **O5** | body 反事实 | `do(width)` → 组内结论是否翻转（body 作为因果算子） | binary_contact + group-flip | **headline** |
| **O6** | 接触定位 | 把「会撞」grounded 到可见首碰证据 | grid | 诊断（可选） |

**O1↔O3 与 d_safe 的对应**
- O1 = `d_safe(r, forward)` 的幅值 / 阈值读出
- O2 = 对 r 反演：`r_crit = max r s.t. d_safe(r, action) ≥ H`，`critical_width = 2·r_crit`
- O3 = `d_safe(r, turnθ→forward) < H`，θ ∈ {±15°(主), ±30°(可选)}
- O4 = `rank over θ of d_safe(r, θ)`，θ ∈ {−15°, 0, +15°}
- O5 = 单题 `label(r)=contact iff d_safe(r,a)<H`；同图同动作不同半径的 `group_id` 内比较符号是否翻转
- O6 = `project(first_contact_3d) → 3×3 grid cell` 或 `none`

### 2.2 读出轴（正交 tag，保证非 MCQ）

| 读出 tag | 答案形式 | 评分 |
|----------|----------|------|
| `magnitude` | 数值，单位 body-width | capped-MAE / bin-acc |
| `threshold` | 会接触 / 不会接触 | Acc / **False-Safe Rate** |
| `ranking` | `right > straight > left` | Top-1 / Kendall τ |
| `binary_contact` + `group-flip` | 单题 contact / no_contact；组内 flip / invariant | **组内翻转 acc** |
| `grid` | `B3` / `none` | grid-acc / 像素距离 |

manifest 里保存标准化 answer schema（HTML 展示页可给自然问法）：

```json
{ "answer_type": "numeric_body_widths", "unit": "body_width", "label": 3.6, "tolerance": 0.75 }
{ "answer_type": "ranking", "items": ["left15","straight","right15"], "label": ["right15","straight","left15"] }
{ "answer_type": "binary_contact", "options": ["contact","no_contact"], "label": "contact", "group_id": "g0427" }
```

### 2.3 Headline 判别器（论文主张的命门）

**O4（跨朝向比较，相对、抗尺度）** 与 **O5（body 反事实翻转）** 是把本 benchmark 与「被伪装的单目 metric depth」拉开距离的**唯一两类**：

- 纯 metric-depth 模型 / center-bias 模型，在 O4 上会因 center bias 失败、在 O5 上**无法翻转**。
- O1/O2/O3 本质上含有「前向距离 ÷ 已知身宽」的成分，与单目深度高度混淆——它们是**地基题（模型是否具备 ego-scaled clearance）**，不是卖点。报结果时 O1/O2 **必须与 center-ray depth 启发式 baseline 并排**，用差距量化「答对里有多少只是深度」。

---

## 3. 相机 / 身体 / 动作配置（v1）

> **height 的定法（修正"抄数"质疑）**：v1 不把 height 当实验变量——唯一被操纵的 embodiment 变量是 **footprint radius**。故 height / 相机位置直接固定为 **Habitat-Sim 默认 agent 配置（height 1.5m / radius 0.1m / sensor 在 1.5m）**：理由是模拟器原生缺省 + 最大可复现 + 前向地面可见性好，**明确不声称等同 ObjectNav LoCoBot(0.88m) 或任何商用平台**。这样既不靠抄某机器人背书，也回避"拼参数"质疑。唯一刻意偏离默认的是 HFOV（90°→79°，**已定**），理由是贴近真实 RGB-D 相机(RealSense ~69–79°)、对"行动前安全"更现实，功能性可辩护。

```text
Simulator : Habitat-Sim + HM3D / MP3D
Embodiment: Habitat-Sim 默认 agent —— height 1.5m, 默认 radius 0.1m, sensor 在 1.5m
            （模拟器原生默认值，不声称匹配任何具体机器人；唯一被操纵的 embodiment 变量 = footprint radius）
RGB sensor: 640×480, camera height 1.5m, HFOV 79°（刻意偏离默认 90°，贴近真实 RGB-D 相机）
Depth     : 同分辨率，仅离线用
Body      : 固定高度 visible cylinder（非纯 footprint），cylinder height = 1.5m（= agent 默认高度）
Radii     : r ∈ {0.10, 0.25, 0.40} m  →  width = 2r（0.10 即默认；其余为同高异宽的 counterfactual 圆柱家族）
Actions   : forward H
            turn-left/right 15° then forward H     # 主
            turn-left/right 30° then forward H     # 可选 stretch tag
forward step granularity : 0.25 m（动作语义），march 步长 0.02 m（oracle）
H (horizon): O1/O3 用 body-width（测 body-relative clearance），{1,2,4,6} body-widths
            【例外】O5 counterfactual 用【固定米/reference-width】，跨 body 同一物理路径（见 §6 红线）
```

**FOV 取 79° 的代价与缓解**：FOV 越窄，`turn30+forward` 的 corridor 越易出框被可见性门拒掉 → 故 O3/O4 的 headline 主用 **±15°**（= VLN `turn_angle`），±30° 降为可选 tag。

> 来源：jacobkrantz/VLN-CE（原版 90°）、Habitat ObjectNav / Habitat-GS（79°）、RealSense D435 ~69°。

---

## 4. Oracle 流程：depth-corridor 主 + navmesh 校验

> 设计决策：**用当前帧 depth 反投影点云定义 label**，而不是用全场景 navmesh。
> 理由：navmesh 用的是「上帝视角」全场景几何，可能拿**当前帧看不见**的几何来判 contact → 产生「单图无法判定」的不公平题。depth-corridor 天然只用可见几何，且**自动给出接触点（O6 免费）**、天然对齐 single-image 公平性。

```text
对每个 (pose, yaw, radius, action):

1. Habitat 渲染当前帧 RGB + depth → 反投影成相机系点云 P。

2. 由 depth 点云构造 obstacle voxel 场 P_obs（不能直接拿原始点云判碰）:
     a. 估计局部地面高度，剔除地面点（floor removal，按离地高度带 + 法向/floor mask）
     b. 剔除 invalid / 远距 / 空洞 depth 点
     c. 剩余可见表面点 voxelize + 1–2 voxel 膨胀（防薄障碍如椅腿被漏）
     d. 垂直高度带过滤（fixed-height cylinder）：离地高度 ∈ [h_min(如 0.05m), cylinder_height=1.5m]
        即把可见表面点投到该高度带内的部分当作圆柱会撞的障碍
     e. 最小支撑：触发碰撞需邻域内 ≥ N 个 voxel，不接受单点

3. 由 action 生成 piecewise swept path:
     forward       : heading = yaw
     turnθ→forward : 原地转 θ（footprint 不变）后沿新 heading 前进
     compound      : 多段 heading 拼接，first_contact = min over 各前进段

4. 沿 path 以 0.02 m march footprint 圆盘(半径 r):
     若 swept 圆盘内 P_obs 支撑 ≥ N:
         first_contact = 当前弧长; 记录接触点 3D 坐标 + 投影像素; stop
     若到 D_max 仍无接触: d_safe > D_max

5. navmesh 独立校验（角色 = hidden-geometry detector + sanity，不是更高真理）:
     per-radius 重新生成 navmesh（NavMeshSettings.agentRadius = r, agentHeight = 固定身高）
     沿同一 path march 中心点 is_navigable，得 d_safe_navmesh
     disagreement taxonomy:
       navmesh 早碰 ∧ contact 不可见 → requires_hidden_geometry=true，丢弃（非 depth bug）
       navmesh 早碰 ∧ contact 在帧内可见 → depth 漏检，review/丢弃
       depth 早碰 ∧ navmesh 未碰 ∧ 无清晰可见障碍 → 点云噪声，丢弃
       depth 早碰 ∧ 可见障碍清晰 → 保留为 visible obstacle case
       两者一致/差距小 → 高置信样本
```

注意：
- per-radius navmesh **必须重新生成**，不要拿默认 navmesh 再额外扣 clearance（否则 double-count radius）。
- compound 轨迹的 GT **不能**由 Habitat rollout「是否提前停止」给出——必须按 piecewise swept volume 计算。rollout trace 只能 debug，不能定义 label。

---

## 5. 四道质量门（缺一不可）

```text
G1 可见性门:
   visible_sweep_ratio ≥ 0.7
   contact 题: 首碰点必须投影落在帧内 且 该像素 depth 有效
   判断若依赖画外 / 遮挡后 / 转角后几何 → 丢弃

G2 no-contact 证据门（硬门，最易造假标签：看不见 ≠ 没东西）:
   no-contact label 必须【整段 horizon 视觉可支撑】:
     visible_sweep_ratio ≥ 0.7
     valid_depth_ratio   ≥ 0.9     # corridor 投影区 depth 有效占比
     depth_hole_ratio    ≤ 0.05
     occlusion_free_ratio 达标      # corridor rays 未被前景遮挡到无法判断
     horizon 终点仍 FOV-supported
   contact label 必须:
     contact point 投影落帧内 + depth 一致性通过 + 支撑点 ≥ N
   任一不满足 → 丢弃

G3 余量门:
   |d_safe - H| ≥ 0.5 body-width            # 阈值题不卡边界
   ranking: top1 - top2 ≥ 0.5 body-width    # 否则丢，或显式标 tie

G4 单调性 sanity:
   r_small < r_large  ⟹  d_safe(small) ≥ d_safe(large)
   （纯 footprint 下「大身体能过、小身体不能过」物理上不应出现；违反即 oracle bug）
```

附加 sanity（数据集级）：
```text
step-size 稳定性: step=0.02 vs 0.01 m 的 label 一致率 ≥ 95%
```

---

## 6. 评分指标（按类别报告，不报 overall-only）

| 操作 | 主指标 | 授权的结论 |
|------|--------|-----------|
| O1 Forward Clearance | capped-MAE / bin-acc | body-relative clearance 是否弱（**对比 depth 启发式**） |
| O2 Critical Width | tolerance-acc / MAE | 是否理解连续 body-width 约束 |
| O3 Turn-then-go Contact | Acc / **False-Safe Rate** | 能否把动作轨迹投进图像做 sweep |
| O4 Directional Ranking | Top-1 / Kendall τ | 能否比较多方向后果 / 是否只有 center bias |
| O5 Counterfactual | **组内翻转 acc**（不是 3-way acc） | 是否把 body width 当 collision operator |
| O6 Contact Grounding | grid-acc / 归一化像素距离 | 是否 grounded 到可见首碰证据 |

**全局最重要指标：False-Safe Rate**（GT 会接触、模型说不接触的比例）——对应「行动前身体安全判断」的现实含义。

**指标细节修正**
- **O5 action horizon 必须是固定物理路径（米 或 reference-body-width），不随各自 body-width 缩放**（红线）。否则 "large 撞、small 不撞" 会被 "large 走更远" 污染，破坏 `do(width)` 因果解释。manifest 存 `horizon_m + horizon_reference=metric_fixed`。
- **O5 题面必须是一题一个机器人**：只描述一个圆柱形底盘直径，二值回答 `contact/no_contact`；反事实翻转只在同 `group_id` 的多 case 之间评估。机器人高度固定为 Habitat 默认 agent height，不写进 prompt。
- O5 主指标 = **窄缝带（r_small no_contact、r_large contact）上的组内翻转准确率**。all-contact/all-no-contact 控制组用于检查 shortcut；用 3-way acc 会让「忽略 body width」的模型虚高。
- O5 配套报三个细分指标（比单一 group-flip acc 更能写 finding）：
    Embodiment Sensitivity = 同一 counterfactual group 内模型答案是否随 width 改变
    Correct Flip Rate      = 是否按 GT 方向翻转
    Invariance Error       = GT 翻转但模型输出完全不变的比例
- O1/O2 数值 = cap 在可见区域内的 capped-MAE + bin-acc（开放空间 >8 身宽会把 MAE 带歪）。
- numeric 容差：absolute error ≤ 0.75 body-width 记为正确。

---

## 7. Baseline 套件（效度门的载体）

```text
必跑:
  random / majority                     → 答案是否平衡
  blind text-only                       → 是否能纯文字蒙
  radius-only / action-only             → 是否靠单一变量
  center-ray depth heuristic            → 【关键】O1/O2 有多少只是深度
  floor/corridor-width heuristic
  geometry oracle（上界）
  MLLM: RGB + question
  MLLM: RGB + sweep-corridor overlay    → 诊断：overlay 后涨很多 = 模型读得懂题但不知 footprint 扫到哪
  human small set
```

**效度门（demo 阶段就要过，否则不进正式版）：**
```text
oracle 自洽（单调性/step-size）          ✓
blind / majority ≈ 随机                  ✓
center-ray depth heuristic ≪ MLLM (在 O4/O5 上)   ✓  ← 立项命门
human > MLLM 且 MLLM 与 oracle/human 有明显 gap   ✓
```

---

## 8. 数据集级 balance / 去重 / 配额

```text
每类单一答案占比不过高（answer balance + FlipEval 思路）
每个 scene 最多占总样本一定比例
同一 RGB 不跨不同 operation 类重复出现
同一 trajectory 最多贡献 1 个非-counterfactual case
counterfactual group 允许同图复用，但必须标 group_id
```

**采样必须分层**（不要随机采完就用）：
```text
scene → floor/island → geometry_tag → answer_label → body_radius → action_angle → difficulty 配额
counterfactual: start pose 必须对 small/medium/large 同时合法
每个起点: 对应 radius navigable、RGB 有足够地面/前方、depth 有效、不贴墙极近、不纯大空地、不强遮挡/画外决定
```

---

## 9. 分析 tag（仅分桶，不做顶层类）

```text
geometry_tag : open-free-space / frontal-barrier / narrow-gap / doorway-threshold /
               asymmetric-side-opening / corner-branch / clutter-near-floor / dead-end
difficulty   : easy / medium / hard（按 margin）
body_radius  : small / medium / large
action_type  : forward / left15 / right15 / left30 / right30 / compound
```

---

## 10. Prompt 设计原则

```text
使用: physical contact / body width / visible local space / from the current view
避免: clear / blocked / safe / dangerous / open corridor / solid obstacle /
      walkable / success / rollout / progress ratio
      （这些词要么泄答案，要么把问题带回导航/affordance）
```

机器人身宽用约值锚定（如「圆柱底盘，宽约 0.5 米」），但 **O1 前向题里不得给「计划距离」**（否则泄答案）。

**prompt 以 body width 为唯一强调的 embodiment 变量，不突出 height**：歧义垂直样本已在 §12 剔除，存活样本的答案基本不依赖精确身高，提 height 反而诱发模型去纠结桌面/横梁等无关推理。height 仅存 manifest 供 oracle 用。

---

## 11. Demo（生死判定）的 DoD

```text
优先 operation: O5 → O4（headline=生死门，先做）→ 再补 O1 / O3
每个核心类 ≥ 30 accepted cases
counterfactual ≥ 30 pairs
≥ 5 个 scenes
同一 RGB 不跨 operation 类重复
visible_sweep_ratio ≥ 0.7
step-size label 稳定性 ≥ 95%
跑 random / majority / blind / depth-heuristic / MLLM / human 六基线过效度门
```

通过效度门 → 决定进正式版（finding 档 vs +method 档）。

---

## 12. 明确不做（hard exclusion，v1）

```text
✗ rollout completion 当 GT（trace 只 debug）
✗ route success / goal reaching / 「能否到达目标」
✗ 楼梯 / 台阶 / 斜坡 / drop-off 作为 v1 核心
✓ v1 = 固定高度(1.5m) visible cylinder：墙/柜/椅腿/箱子/门框侧 等离地到身高带内、可见稳定的垂直障碍【收】
✗ 丢弃 footprint-vs-cylinder 会分歧的歧义样本：overhang / 桌面下钻 / 接近 cylinder_height 临界的台面边缘
   （这类答案依赖"基座能否从下方穿过"，对固定高度圆柱判定不稳，剔除）
✗ 遮挡后 / 转角后 / 画外区域
✗ prompt 出现泄答案的场景描述
✗ LLM 生成 image-backed label
```

v2 再扩展：不同 camera/body height、low robot vs humanoid、垂直 3D body envelope、profile-specific re-render，核心函数扩成 `d_safe(agent_profile, action)`。

---

## 13. 与相关 benchmark 的区隔（已核验 prior art）

| 工作 | 它做什么 | 我们的区隔 |
|------|----------|-----------|
| PhysBench (ICLR'25) | physical world understanding（object 属性/关系/动力学） | 禁用 "physical world understanding" 措辞；我们是 ego-body action consequence |
| CapNav (CVPR'26) | capability-conditioned **路线级**导航（tour video + nav graph + mobility profile） | **最危险近邻**；我们无 route/goal，只问单图单动作局部后果 |
| TouchSafeBench (2605.31196) | collision grounding，多视角 RGB-D，看**当前/即将**接触 | 我们是单张 RGB 下**假设性**动作的 swept-volume 后果 |
| 3DSRBench / Spatial457 / VSI-Bench | 3D 空间推理 / 6D / egocentric video 空间智能 | 借鉴「离线几何生成 + 前台只给视觉 + answer balance」范式 |

可活楔子：**single-image + counterfactual ego-body sweep + 仅可见区域**。

---

## 14. 决策记录（多轮讨论收敛）

1. **分类轴**：操作轴×读出轴两根正交轴，O4/O5 定为 depth-robust headline。（拒绝「保持 C1–C6 原样」，因混轴 + C1≈C2-fwd 注水 + C2 自身混 forward/turn 两种操作。）
2. **GT oracle**：depth-corridor 主（定义 label，单图公平 + O6 免费）+ navmesh 独立校验 + 深度空洞门。（拒绝 navmesh 为主，因其用画面外几何判 contact = 不公平。）
3. **相机 FOV**：79° / 640×480（对齐 Habitat ObjectNav，比 90° 更贴真实机器人）；headline 转向主用 ±15°，±30° 可选，以对冲窄 FOV 对 O3/O4 产量的挤压。
4. 本文件升 v1.0 冻结后，下游生成器从中派生，**不得写死常量空转**（吸取上一版生成 agent 的最致命教训）。

### v0.9 一轮外部 review 后的修订（2026-06-30）
- 状态由"冻结"降为 **v0.9 candidate**（oracle 物理细节定下来前不 final freeze）。
- **O5 horizon 改为固定物理路径**（红线，原按各自 body-width 缩放会污染 do(width) 因果）。
- **depth predicate 重写**：去地面 / voxel 膨胀 / 最小支撑，禁单点触发（原 [0,body_height] 含地面会假碰撞）。
- **navmesh 校验改为 disagreement taxonomy**（hidden-geometry detector + sanity，非硬比较）。
- **G2 升级为 no-contact 证据门**（valid_depth_ratio≥0.9 等）。
- **§3 相机/身高配置**：height 不是 v1 实验变量（只动 footprint radius），故固定为 **Habitat-Sim 默认 agent 1.5m/0.1m**，如实声明"模拟器默认、不匹配任何机器人"——而非借 ObjectNav/ImageNav 的数背书（一度误选 0.88m"为干净引用"，已纠正）。
- **Demo 顺序**：O5/O4 前置为生死门；O1 能被 depth 解出≠失败。
- O2 重定位为 body-width 家族连续诊断（不与 forward/turn/ranking 并列为独立操作）。

### OPEN-1 已决（2026-06-30，连带 §3/§4/§10/§12 已写定）
选定 (a) **固定高度 visible cylinder + 丢弃歧义样本**：footprint-only 与固定身高物理不自洽，故 v1 用固定高度圆柱碰撞、剔除 overhang/桌沿临界等 footprint-vs-cylinder 分歧样本。
**height/相机 = Habitat-Sim 默认 agent（1.5m / radius 0.1m / sensor 1.5m）**：因 height 在 v1 是held-constant 无关量（只操纵 footprint radius），固定到模拟器默认即可，如实声明不匹配任何商用机器人 → 不靠抄数背书、且回避拼参数质疑。（曾误选 0.88m 想"干净 cite ObjectNav"，被指出仍是借身份背书，已纠正回默认 1.5m。）
连带：cylinder_height=1.5m → §4 垂直带上界=1.5m；prompt 只提 width；§12 由"不做垂直 envelope"改为"收可见稳定垂直障碍、丢弃歧义"。
HFOV=79° 是唯一刻意偏离默认(90°)处，功能性理由(真实 RGB-D 相机)，**已定 79°**。

---

## 附：待办（落地顺序，最薄 vertical slice 优先）

> 遵循「先搭最薄 end-to-end skeleton + 每加一层一个 smoke gate」。

```text
P0  1 个 Habitat 场景 + 1 个 pose: 渲染 RGB+depth → 建 obstacle voxel(去地面/膨胀/支撑) → d_safe(forward)
    smoke: 人工看一张图 + overlay + d_safe 数值是否合理
P1  接上 navmesh 校验(disagreement taxonomy) + 单调性 sanity
P2  【生死门·先做】O5 narrow-gap group: 验证 small no_contact / large contact，且 horizon 固定物理路径
    gate: 同图换 width，GT 翻转，MLLM 是否翻转？(headline 1)
P3  【生死门】O4 directional fan: 验证 left/straight/right 有明显 d_safe 差异
    gate: depth/center-ray heuristic 是否 ≪ MLLM？(headline 2)
P4  再补 O1(magnitude) / O3(turn)。注意: O1 能被 depth heuristic 解出≠失败，O1 只是地基题，不当生死门
P5  扩到 ≥5 scenes，跑满六基线 + 四道效度门 → demo 生死判定
```
