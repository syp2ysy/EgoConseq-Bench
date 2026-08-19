# EgoConseq：我们到底在测什么——任务体系设计与分类根据

日期：2026-08-10
状态：**设计根据文档（design rationale）**。
与其他文档的关系：`2026-08-03-abc-first-order-consequence-freeze-design.md` 仍是六个已冻结任务
（A1/A2/A3/B1/B2/C1）语义的唯一规范来源；本文不改变它们的语义，负责回答三个问题——
benchmark 整体测什么、为什么这样分类、每个子任务问什么/测什么。本文中标注
**[已冻结]** 的内容以冻结稿为准；标注 **[修复]** 的是不改语义的实现修正；标注
**[amendment 候选]** / **[scan 后定]** 的是本文提出、尚未走冻结修订流程的新增项。

---

## 1. Benchmark 测什么：一句话与三根支柱

> **EgoConseq 测试一个视觉语言模型能否把「一张标定的单张第一视角 RGB + 一个显式的
> 圆盘身体（半径公开）+ 一段米制 SE(2) 动作程序」转化为「由初始可见空间所决定的、
> 可机械验证的执行后果」。**

这句话里有三根支柱，每根支柱对应一个公开杠杆：

| 支柱 | 含义 | 对应杠杆 | 杠杆的签名 |
|---|---|---|---|
| ① 身体占据 | 机器人不是一个点，是半径 r 的圆盘；扫掠面积、余隙、可通过性全由 r 参与决定 | do(body radius) | 半径变 → 碰撞标签在临界半径处翻转 |
| ② 动作几何 | Forward(米)/Turn(度) 的 SE(2) 合成；顺序有意义、数值有意义 | do(action) | 程序变 → 全部后果正确地变 |
| ③ 可见空间 | 单张图定义了知识边界；答案必须由可见几何决定，模型也必须知道这条边界在哪 | do(FOV / 相机高度) | 观测变 → 物理答案逐字不变、观察答案正确地变 |

**后果只是测量仪器。**每个子任务存在的唯一理由是：它逼模型使用三根支柱中的至少一根，
且以当前任务集尚未隔离的方式使用。这是全文的审判标准。

战略定位（motivation 的应用层表述）：**执行前后果验证器（pre-execution consequence
verifier）**。2026 年的 VLA 安全评测（SafeVLA-Bench 等）全部通过 rollout 事后统计策略的
碰撞率——没有任何 benchmark 测量"感知模型能否在执行之前、从一张图预判一段程序的后果"；
导航前瞻模型（NavWAM/FutureNav/UniWM）在训练视觉想象模块，但想象是否几何正确从未被
认证 GT 审计过。EgoConseq 的任务集就是这台缺失的仪器：碰撞（A）、余量（A4）、终点
（B）、未来观察（C）恰好是安全类指标（如 NAVSIM PDMS）聚合的全部分项，而我们把每一项
变成带认证 GT 的感知探测。

---

## 2. 为什么这样分类

### 2.1 分类原则：读出轴按因果阶段，协议轴按测量设计

一次执行只产生一个客观对象——**结构化后果记录**：

```
                    ┌─ 碰撞分支:  (collision, 第k个动作, 接触实体e, [触点方位, 截断进度])
s0 + body + action ─┤
                    └─ 安全分支:  (clear, [擦过状态], 终点位姿 → 目标距离d, 目标方位θ, 终点画面oT)
```

**任务头 = 这个记录沿因果链的读出位置**，不是按答案的表面类型（"物理题/语义题/视觉题"）分：

```
initial state + action
        │
        ▼
 execution trajectory ───────► A · 过程读出（接触与贴近事件）
        │
        ▼
 realized clear endpoint ────► B · 终态物理读出（与地标的度量关系）
        │
        ▼
 terminal camera render ─────► C · 终态观察读出（相机会看到什么）

 （横切）初始视图的知识边界 ──► E · 认知读出（这个后果从此图可判定吗）
```

与读出轴正交的是**协议轴**：同一道题可以在自然采样下问，也可以在受控干预对/三元组里
问。协议不产生新任务 ID，只产生新的诊断指标（见 §7）。**曾经的"D 类"就是把协议误当成
类别的教训**：状态等价三元组问的仍然是 A/B/C 的问题，因此它不是第四个头，而是第四个
干预算子 do(history|state)——现已降格为协议，冻结稿的六任务清单因此无须为它改动。

### 2.2 头与头真正可分的证据：干预签名

三个杠杆恰好把三个读出头分开——这是"A/B/C/E 是真实的不同类别"的最硬论证：

| | A 过程 | B 终态物理 | C 终态观察 | E 认知 |
|---|---|---|---|---|
| do(action) | 变 | 变 | 变 | 变 |
| do(body) | **变**（A1/A4a 翻转） | 仅经由可行性 | 仅经由可行性 | 仅经由走廊宽度 |
| do(sensor) | 必须不变 | 必须不变 | **必须正确地变** | **必须正确地变**（FOV 变 → 可见空间变 → 可判定性可变） |

### 2.3 完备性：读出空间是被穷举证明的，不是拍出来的

在冻结 scope（静态世界、确定性开环、首次接触截断、单张初始 RGB、答案由可见几何决定、
闭式机械 GT）内，我们对候选读出做过 28 项穷举，三条归约引理杀掉绝大多数：

1. **反解引理**：终点位姿（航位推算）⊕ (d, θ) 可精确反解目标世界坐标 → 一切"目标点
   × 轨迹"的量（最近逼近、目标何时离开视野、屏幕坐标、视大小变化、接近性）都是已有
   读出加公开参数的算术；
2. **实例参数引理**：marker 和程序是已有读出类型的合法输入 → 多物体比较、左右序、
   最近实体、双程序比较、逆向程序辨识全部归约为"已有类型的多次实例 + 比较"；
3. **圆盘引理**：Turn 不产生新扫掠面积 → 终端朝向、碰撞动作类型等退化为纯程序算术。

幸存的非冗余读出共 5 个家族，本文全部给了归属：截断进度（→A7）、接触构型（→A6）、
间隙家族（→A4a，"临界半径"的反事实问法被否决，改为过程问法）、终端视线遮挡（→C2）、
认知覆盖（→E2）。**因此"任务是否足够"是一个有证明的问题：在声明的 scope 内，本文的
任务集覆盖了全部非冗余读出；scope 外缺失的一整类（接触后世界状态变化）见 §9。**

### 2.4 A 头的对称结构：两个分支、同四个坐标

一次执行的时间线必有一个**关键事件**：碰撞分支是**首次接触**，安全分支是**最近逼近**
（最险的擦过）。当前设计把安全分支的过程读成了零分辨率——贴 2 cm 擦过和两侧各余 1 m，
A1 给同一个标签。A4 家族补的就是这一半，且与 A2/A3 完全对称：

| 坐标 | 碰撞分支（首次接触） | 安全分支（最近逼近） |
|---|---|---|
| 发生了什么 | A1 = collision | A1 = safe → **A4a 惊险/从容** |
| 何时 | A2 第几个动作接触 | **A4b 第几个动作最险** |
| 何物 | A3 接触了什么 | **A4c 离什么擦得最近** |
| 何种程度 | A7 撞停在动作的前/中/后段（可选） | （已并入 A4a 二值化） |

条件级联同构：`collision → A2/A3`；`safe → A4a`；`near_miss → A4b/A4c`。

### 2.5 三支柱覆盖矩阵（分类的最终自检）

| 任务 | 身体占据 | 动作几何 | 可见空间 |
|---|---|---|---|
| A1/A2/A3 | ✓（临界样本+paired core 才隔离） | ✓✓ | 仅作数据门 |
| A4 家族 | **✓✓ 隔离余量** | ✓ | 仅作数据门 |
| B1 / B2 | ✗ | ✓ 平移 / ✓✓ 旋转 | ✓ 单目度量深度 / ✗ |
| C1 / C2 | ✗ | ✓ | ✓ 视角变换 / ✓✓ 遮挡几何 |
| **E1 / E2** | ✗ | ✓ | **✓✓ 知识边界本身** |

没有 E 之前，第三支柱（可见空间）在整个 benchmark 里的身份只是**数据资格审查员**——
证书拿它筛数据，不合格就扔，但从没有任何一道题要求模型自己判断可见性的边界。
"初始可视空间内"曾经只是我们对数据的约束；E 头把它变成对模型的测试。

---

## 3. A 头 · 过程读出：这段动作能不能安全执行完、执行时发生了什么

**头的意义**：身体扫掠走廊与环境自由空间的冲突判定。核心能力 = 把「半径 + 米制动作」
合成一条扫掠走廊，与图中可见障碍求交，并给出交互事件的完整坐标。

### A1_collision **[已冻结]**

- **题面**：Will the robot collide with anything while executing all the actions?
- **答案**：`collision / no_collision`
- **测什么**：走廊与障碍**是否相交**（存在性判定）。这是唯一强制"路径感知"的基础题——
  只算终点、无视路径的模型（teleportation 错误）在中途碰撞样本上必然失败。
- **答错暴露**：走廊合成错误或余隙判断错误。
- **GT**：身体条件化 navmesh 完整扫掠 rollout + 初始 depth 可见空间共识（双 oracle）。

### A2_collision_step_grounding **[已冻结]**

- **题面**：The robot will collide while executing the actions. During which action does
  the collision occur?
- **答案**：`action_1 … action_N`（由公开动作数决定，**保留 Turn 选项**——Turn 作为
  永不正确的选项本身在测"模型知不知道圆盘原地旋转不产生新扫掠面积"）。
- **测什么**：接触事件在弧长上的**时间定位**。
- **计分纪律 [修复]**：同时报完全随机基线 1/N 与"知道 Turn 不可能"的 informed 基线
  1/F（F=Forward 数）；按 first/middle/last Forward 配平。
- **答错暴露**：A1 对而 A2 错 = 知道会撞但弧长积分错。

### A3_contact_object **[已冻结，一处 open decision]**

- **题面**：The robot will collide while executing the actions. What category of visible
  object or surface will it contact first?
- **答案**：初始可见、可接触类别的确定性去重列表（3–11 项，chance 逐题不同，必须报
  chance-adjusted 与按选项数分层的成绩）。
- **测什么**：接触事件的**实体归因**。
- **诚实边界（open decision）**：当前实现只验证类别归因——路径上有 chair、远处有
  cabinet 时，凭大致方向答 chair 即可，不要求定位接触点。二选一必须择其一：
  (a) 论文中如实命名为 first-contact **category attribution** 并收窄声明；
  (b) 升级为 marker 实例指认 + corridor-near 干扰项门（几何上可能被撞到的错误类别
  才能进选项）。若选 (b)，A4c 与 C2 的 grounding 约定必须同步（三者共用一套 marker
  协议，不允许一半类别一半 marker）。

### A4a_near_miss **[amendment 候选]**

- **题面**：The robot completes all actions safely. Does it pass within 5 cm of any
  obstacle at any point, or does it keep comfortable margins throughout?
- **答案**：`near_miss / comfortable`
- **测什么**：**余量感知**——"过得去"和"贴着过"的区别。这是安全语义上最有价值的一档
  信息（等价于我们设定下的 time-to-collision 类指标），也是身体支柱最直接的题面级
  落点：5 cm 的判断依赖于知道自己有多宽。
- **GT（纯复用现有机制）**：`near_miss` ⟺ 半径 r+0.05 的 rollout 碰撞；`comfortable`
  ⟺ 半径 r+0.10 的 rollout 仍安全；两者之间的样本丢弃（迟滞带，解决余隙贴阈值时的
  标签抖动）。逐半径 navmesh 重建的非单调 artifact 样本整体丢弃并统计弃样率。
  数学基础：路径最小余隙 < δ ⟺ 半径放大到 r+δ 后碰撞——"过程中的擦过"与"身体反事实"
  是同一信息的两种问法；**采用过程问法**（"这次执行有多险"是后果；"换个半径会怎样"
  跳出了"这台机器人执行这段程序会发生什么"的叙事框架，已否决）。
- **捷径控制**：50/50 配平；near_miss 与程序长度/复杂度天然相关 → 按程序统计分层
  配平 + program-only 盲基线。
- **数据来源**：与 body paired core（路线 A）同一批多半径 rollout，零额外采集。

### A4b_tightest_action / A4c_nearest_entity **[amendment 候选，条件于 A4a=near_miss]**

- **题面**：(b) During which action does the closest approach occur? /
  (c) Which visible object does the robot pass closest to?
- **测什么**：最近逼近事件的时间定位与实体归因（A2/A3 的安全分支镜像）。
- **门（平移自 A2/A3 现有门）**：最近逼近点须严格落在某个 Forward 内部、距动作边界留
  margin（注意 Turn 期间圆盘不动、余隙恒定——若最小值恰好在"Forward 末端贴墙随后原地
  转"的位姿上取得，它属于两个动作的边界，此类样本弃）；最近实体领先第二名 ≥ Δ 防
  tie；见证表面初始可见（同 A3 contact witness）；扰动下归因稳定。

### A6_contact_side / A7_arrested_travel **[可选，最低优先级]**

- A6：接触发生在机身的左前/正前/右前（接触构型，穷举幸存者）。
- A7：已知在第 k 个动作碰撞，撞停在该动作的前段/中段/后段（粗分档避开米制精度问题；
  GT 免费——接触弧长本来就是 A2 稳定性门每题在算的量）。

---

## 4. B 头 · 终态物理读出：安全走完后我在世界中的哪里

**头的意义**：终点位姿与一个初始可见地标的**度量关系**。核心能力 = 航位推算（动作合成）
× 单目度量感知（把图中目标 ground 到米制三维），缺一不可。

**为什么恰好两个子任务**：B1 + B2 = 目标在终点自身坐标系下的极坐标 (d, θ)，对"终点—
目标"这个二元关系是最小完备读出；穷举已证明目标相关的其他一切候选（屏幕位置、视大小、
是否在视野内、最近逼近……）都是 (d, θ) + 公开参数的算术。两者失败模式天然正交：
**B1 吃深度/尺度，B2 吃转角积分**——一个模型可以 B2 全对而 B1 全错（会转向但没尺度感），
这个分离本身就是诊断。

### B1_endpoint_distance **[已冻结 + 三项修复]**

- **题面（修复后）**：After safely completing all the actions, what will be the
  horizontal distance **from the center of the robot's circular footprint** to the
  nearest ground-level part of {target}?
- **答案**：4 个米制选项（间距 0.25 m）。
- **测什么**：终点径向坐标——米制尺度感。
- **三项修复（不改 GT 锚点语义）**：
  1. **rank 均匀化**：旧干扰项构造（truth±k×0.25 哈希选 3）使正确值落在四个数值中间
     两位的概率达 90%（实测 26 题 81%），"盲猜中间值"基线 45% ≫ 名义 25%。修法 =
     先均匀采样正确值的数值 rank，再生成干扰项；配平数值 rank × 展示位置 × 距离桶。
  2. **参考点写死在题面**：GT 是机器人中心到目标 ground support 的最近水平距离
     （不减半径，因此不是 clearance，不得称 clearance）。原题面只说 "the robot"，
     而 body_radius 是公开输入——按车身边缘作答的模型系统性掉一格，且这个歧义的方向
     恰好惩罚"认真使用了身体尺寸"的模型，与 benchmark 的核心主张自相矛盾。
  3. **visible-near-face 证书**：终点处完整 support 的最近点必须落在初始可见 support
     的 τ 邻域内，否则 withhold——堵"隐藏背面几何决定答案"（实测 20 终点中 6 个的
     可见/完整差超过一个选项档，最大 1.42 m）。完整几何仍是 GT 权威；证书只筛样本。

### B2_endpoint_direction **[已冻结 + 一项证书]**

- **题面**：After safely completing all the actions, where will {target} be relative to
  the robot's final facing direction?
- **答案**：`front / left / right / rear`（四扇区，边界 margin 15°）。
- **测什么**：终点角坐标——转角积分与朝向合成。
- **修复**：**sector 不变性证书**——由初始可见表面构造的可见质心 bearing 与完整表面
  质心 bearing 必须落同一扇区（现有 15° margin 只保证完整质心不贴边界，不保证隐藏
  几何引起的质心偏移不翻扇区；实测偏移最大 35.7° > 15°，虽然该样本未翻转，需要证书
  而不是运气）。B1 的最近点锚与 B2 的质心锚是两个分别冻结的锚点，不合并。

---

## 5. C 头 · 终态观察读出：安全走完后我会看到什么

**头的意义**：把推演出的物理状态**翻译回观察空间**——完成「观察 → 世界模型 → 新位姿
→ 新观察」的整条闭环。C 与 B 的形式区分不是"视觉 vs 物理"，而是干预签名：改 FOV/相机
高度，B 逐字不变、C 必须正确地变。**C 是唯一被 do(sensor) 杠杆作用的头。**

**为什么恰好两个子任务**：整幅观察（C1，一元：我 × 全场景）与视线穿插（C2，三元：
我 × 物 × 物）互补对方的盲区——C1 可以靠全局外观匹配答对而不理解深度层次；C2 恰好把
深度层次单独拎出来，且是 B1+B2+内参推不出的唯一目标级量（无论把目标的距离方向答得多
准，都推不出"视线中间有没有东西"）。穷举证明这两者之外 C 没有第三个非冗余读出。

### C1_future_view_selection **[已冻结 + 协议与干扰项升级]**

- **题面**：After safely completing all the actions, which image shows the robot's
  final view?
- **答案**：4 张候选终点渲染图之一。
- **测什么**：动作 → 终点位姿 → **整幅**第一视角观察的完整映射；要求同时算对位移、
  朝向，并把场景三维布局重投影到新视点。
- **三个协议（不新增任务 ID）**：
  - **C1-Natural**：自然采样，估计真实难度与错误分布；
  - **C1-Action-Contrast**（冻结稿 §6 已要求、未实现的欠账）：同一初始图 + 同一候选
    集 + 两条统计匹配（同总前进/同 primitive 数/同 net yaw）但顺序不同的程序，正确图
    必须翻转；指标 C1-PairExact；
  - **C1-Sensor-Contrast（核心协议）[amendment 候选]**：同 pose/body/action，发布两个
    sensor cell（79°/110° 或不同相机高度）。物理零重跑，只重渲染 + 按各自 FOV 重算
    可见空间证书（**以窄 FOV 为准**：相关几何只在宽 FOV 可见则整对 withhold）。
    产出两个别人给不出的数字：**SensorInvariance**（换 FOV 后 A/B 答案是否逐字不变——
    失败 = 模型把"看得见"当成"存在"）与 **SensorSensitivity**（两个 cell 的 C1 都对）。
    必配控制：110° 角分辨率更低，须并排报 per-cell 一阶准确率以分离难度效应。
- **干扰项升级 [修复]**：干扰项从"匹配的 sibling"升级为**具名误差模型**——每个错误
  候选标注它对应的推理错误（`ignored_final_turn` / `turn_sign_flipped` /
  `order_permuted` / `rotation_only` / `translation_only`）。选错哪张图 = 犯了哪种
  错误，单个准确率变成一张错误剖面；同时结构性杀死最近邻相似度基线（所有干扰项都是
  近邻）。诚实声明：四选一测的是 discrimination 不是 generation，论文明写。

### C2_terminal_occluder **[scan 后定]**

- **题面**：After safely completing all the actions, which marked object, if any, will
  block the robot's line of sight to target T?
- **答案**：`clear / A / B / C`（marker 恒 3 个，字母随机分配）。
- **测什么**：终点视角下「相机—遮挡物—目标」三点的**射线穿插关系**（深度层次）。
- **GT**：从认证终点相机向 T 的初始可见表面射线投射；≥60% 被挡 = occluded、≤10% =
  clear、中间弃；胜出遮挡者占被挡射线 ≥80% 且属于 marker 集；位姿 + 采样种子扰动下
  满足迟滞（≥70% / ≤5%）。
- **五条发布门（按重要性）**：
  1. **同图反事实程序组（生死门）**：同一初始图 + 同一 marker 布局配多条程序，组内 GT
     覆盖 clear 与不同字母且均衡。一条规则同时杀死四类捷径：clear 频率先验（自然分布
     60–80%）、**初始帧居间性启发式**（"初始图里压在通往 T 连线上的物体"——这条捷径
     恰好冒充任务要考的能力，不杀等于白做）、语义先验（大件才挡光）、marker 位置先验。
     配套硬指标：用**初始位姿**做 ray-cast 的 oracle 基线必须 ≈ 随机。
  2. 背面遮挡门：真正挡光的面必须初始可见，且该可见面投影到终点视角覆盖被挡区域
     ≥70%（否则挡光的是柜子背板，单图原则上不可解）。
  3. FOV 分离门：T 的无遮挡投影必须在终点 FOV 内留 margin；out_of_fov 样本整体排除
     （出画框是 B2+HFOV 的算术，不许混进遮挡题充数）。
  4. mesh 完整性门：射线走廊内经过开放边界 ε 邻域即弃（MP3D 孔洞是系统性的）；细杆、
     镂空、透明、镜面类黑名单。
  5. marker 卫生：贴初始可见 mask 质心、字母随机重排、marker-only 探针 ≈ 随机。
- **准入条件**：feasibility scan 逐门存活率 + 人类小样本上限 ≥85%，两者都过才提交
  amendment；61 个 R2R 场景撑不起同图配平则 C 正式只有 C1，不硬上。
- **prior-art 定位**：novelty 收窄为"在未观测、仅由动作程序定义的终点位姿上做视线
  遮挡者 grounding + 几何可验证 GT"；World2VLM D2（metric 动作 + 二值可见性，不分
  出框/遮挡、无归因、非 benchmark）与 CrossView Suite Q8（遮挡者归因但所有视图给定）
  是必须正面处理的两翼。

---

## 6. E 头 · 认知读出：这个后果从此图可判定吗 **[amendment 候选]**

**头的意义**：可见空间支柱的**模型侧读出**。模型必须知道自己知识的边界——把"看得见"
和"存在"分开，把"没看到障碍"和"没有障碍"分开。没有 E，"初始可视空间内"只是论文修饰语。

**为什么它能成立（关键构造）**：E1 的"不可判定"不是人工标注，是几何判定——**扫掠走廊
穿过初始视野的遮挡区，且盲区体积足以容纳一个最小尺寸障碍物 ⟺ 两种结局都与可见几何
相容 ⟺ 不可判定**。对象问题（撞不撞）的答案依赖隐藏几何，但**元问题（能不能判定）的
答案由可见几何完全决定**——E1 不违反"答案必须由可见空间决定"的冻结原则，它恰恰是这条
原则本身变成了题目。GT 原料 = 证书体系当前当废料扔掉的 reject 流（本次 smoke 中仅
`future_view_evidence_insufficient` 一项就弃 51 个 outcome）。

### E1_outcome_determinability

- **题面**：Given this view, this body, and these actions — will the robot collide?
- **答案**：`no_collision / collision / cannot_be_determined_from_this_view`
- **测什么**：可见性边界的判断——模型是否知道"这张图不足以回答这个问题"。
- **为什么重要（三层）**：① motivation 层：三支柱补全；② 安全层：可部署的执行前
  验证器必须会说"我看不出来"——在不可判定样本上硬答 no_collision 的模型正是部署中最
  危险的模型；false-safe rate 由此升级为"错误 + 过度自信"的完整安全画像；③ 护城河层：
  认证的不可判定 GT 需要可见空间证书体系才能生产，这套设备只有本 pipeline 有。
- **捷径控制**：三类答案配平；"不可判定"与程序长度/转向统计天然相关 → program-only
  盲基线 + 刻意构造短程序盲区样本（近处大遮挡物，两步进盲区）+ 表面统计分布匹配。
- **prior-art（已核验，槽位开放但窗口收窄）**：无任何工作同时具备（i）动作条件碰撞
  语义、（ii）三值 epistemic 答案、（iii）几何认证的不可判定标签。最近邻及切割：
  SpatialUncertain 2605.30557（唯一构造式认证 unanswerable 空间 QA，但物体级非体积级、
  无碰撞/动作程序；必引最强 prior art）；Spatial457 2502.08636（唯一单图 will-it-collide
  VQA，但二值且 GT 来自全知场景图——被遮挡物照样计入答案，恰好示范我们要修的缺陷，
  直接用作 motivating example）；VB 2603.06680（三值 + OCCLUSION 理由码但人工 GT——
  novelty 泛化成"知道自己看不见什么"会被它压住）；AbstainEQA 2512.04597（"embodied
  弃答"旗帜但不可答性来自问题病态——泛化成"embodied abstention"会被它压住）；
  Yes-Man 2605.20544 / VLN-NF（**指令级**可行性 abstain，与我们的**视图级**几何证据
  不足必须显式切割，审稿最易混淆处）；VG-AVS 2512.13250（构件级重叠最大：单图 + 米制
  视点动作 + 仿真 GT，但评策略成功率，模型从不读出可知性）。novelty 陈述钉死在
  **"几何证书 + 动作程序条件化"**两点。

### E2_blind_segment **[scan 后定]**

- **题面**：During which action does the robot first enter space that cannot be seen
  from the initial view?
- **答案**：`action_1 … action_N`
- **测什么**：可见性边界的空间定位（E1 之于 E2 = A1 之于 A2）。与 C2 同批 scan。

---

## 7. 协议轴：四个干预算子（不是任务，是"怎么测"）

| 算子 | 固定什么 / 改什么 | 签名预测 | 装置与指标 |
|---|---|---|---|
| do(action) | 同 s0/body/sensor，只改程序 | 后果正确地变 | action-contrast 对（统计匹配、只差顺序）；C1-PairExact |
| do(body) | 同 s0/action/sensor，只改半径 | A1/A4a 在临界半径翻转 | **body paired core（路线 A）**：同 s0 重跑三半径，保留转变样本，验单调性 y(0.15)≤y(0.20)≤y(0.25)；指标定义在**转变位置**上（防"最小=安全、最大=碰撞"白拿 2/3） |
| do(sensor) | 同 s0/body/action/物理，只改 FOV/高度 | A/B 逐字不变、C 正确地变 | sensor-contrast 对；**FOV-leak rate**（A/B 答案随 FOV 改变的比例 = 模型把观测范围误当世界状态的量化） |
| do(history\|state) | 同中间状态/续段，只改到达路径 | 全部后果不变；统计匹配但状态不同的对照必须正确地不同 | **状态等价三元组**（原"D"）：模型只见三道平铺长程序普通题（无 Segment 字眼），结构只活在 evaluator 的 family 元数据与联合计分里 |

状态等价三元组的规范要点（红队三轮后的终版）：

- **构造**：废除纯镜像 dogleg（终点被对称性钉死在正前方轴 / 最大转角位置 tell / 首转侧
  = 漂移侧，三重表面泄漏）。改三族混用——**构造 B forward-redistribution**（转角序列
  逐字相同、前进多重集相同，仅重分配：`L25-F0.70-R45-F0.40-L45-F0.20-R25` 与
  `L25-F0.20-…-F0.70-R25` 同端点，`L25-F0.40-R45-F0.70-…` 为异端点对照——表面统计
  通道整个焊死）+ 构造 A（逆序排列对）+ 构造 C（跨形状对）。
- **三张证书**：answer-level 等价（碰撞是路径属性——pair 前段走廊各自 clearance ≥
  margin，使碰撞只可能发生在共享续段，否则碰撞题不入 pair 计分）；contrast bin 分离
  （对照真值与 pair 真值隔 ≥1 选项档且扰动稳定）；反 surface（对幅值序列/最大转角位置/
  首转侧/步长排列训练的探针识别对照程序的 AUC ≈ 0.5）。
- **计分**：唯一 headline = **隔离呈现下的 triple-exact**（联合呈现允许模型复制自己的
  答案）；invariance 永不单独报（转向符号盲/前段盲的模型天然满分——该指标与能力反相关），
  只报 contrast-conditioned 版本；掺入模板破坏样本使 chance 可实测；三个程序化消融
  （删前段/前段换净位移/符号打乱）作为公开 floor，删前段即可全对的三元组直接剔除。
- **归因探针 D0**（不上榜）：P1/P2 是否停在同一位姿——区分"连位置都算不对"与"位置对了
  但后果预测没有经过位置"，只有后者是协议要抓的失败。

## 8. S-track：结构化无条件预测（capstone）

原子任务的题面全部泄漏了分支条件（A2 明示"会碰撞"、B/C 明示"安全完成"）——模型从不需要
自己判断分支，测到的是 teacher-forced 条件级联。S-track 一次性无条件输出完整后果记录：

```
collision?
├─ yes: 第几个动作 / 接触什么 / (机身哪侧)
└─ no:  离目标多远 / 什么方向 / 途中是否擦过 / 哪张是终点画面
```

指标 **record-exact**（整条记录全对）是全 benchmark 最硬的数字，直接测"模型能否自主
维护一个自洽的后果状态"。GT 零新增，只需提交格式与 evaluator。

## 计分体系（headline 与诊断的分界）

- **Headline**：A/B/C/E 四头原子均分（等头权）+ **false-safe rate**（安全语义主指标）
  + S-track record-exact。
- **诊断列（不入 headline）**：四算子协议指标（PairAcc / BodyPair 转变位置 /
  FOV-leak / triple-exact）、C1 错误剖面、A1–A4a coherence、E1 弃答质量。
- **统计纪律**：item-micro 与 s0-macro 并报；scene 聚类 bootstrap CI；同 s0/rollout/
  family/前缀蕴含关系进同一 split（A5 类多半径题与 A1 互相蕴含，必须同 conflict-graph
  分量）；每任务并报全部盲基线（answer-frequency / position / program-only /
  image-only / 各任务专属捷径基线）。

---

## 9. 明确排除项（及为什么）

| 排除 | 理由 |
|---|---|
| **接触后世界状态变化**（推动/倾倒/开合/破损） | 后果本体四类（agent 状态/世界状态/关系/信息）中缺失的一整类，但静态扫描底座无质量/摩擦/铰接 → GT 不可认证；用语义先验（chair=可推）编标签会摧毁"答案由认证 simulator 机械生成"这条立身之本。写成显式 scope 边界 + **v2 路线图**（需物理底座），不在 v1 硬塞 |
| 动力学 / 他人反应 / 声触觉 | 同上，底座外 |
| planning / best-action / 可达性 | 冻结稿 §8 明令排除；"哪个动作更好"把 benchmark 变成策略评测 |
| 目标屏幕位置 / 视大小变化 / 是否在画框内（作为独立任务） | 冻结稿明令废除且被反解引理证明冗余（u=f·tanθ+cx；面积比=(d0/dT)²） |
| closer/farther 独立子题 | 冻结稿明确不设（B1 距离变化的符号） |
| "最大安全半径"反事实问法 | 信息 = A4a 的 body-grid 阈值化，但措辞跳出后果叙事；已由 A4a 过程问法承载 |
| 闭环 / 中途给图 | 破坏单图契约，ABC 全部证书体系需重写——那是另一个 benchmark |
| 通过侧 / 最近物体比较 / 视野重叠比例 | 不隔离任何支柱（分别只用已充分覆盖的动作几何、泛空间整合、粗全局量）——按 §1 审判标准撤回 |

---

## 10. 全景与落地顺序

**最终任务集**：

| 头 | 任务 | 状态 |
|---|---|---|
| A 过程 | A1 / A2 / A3 | 已冻结（A3 有一处 open decision） |
| | A4a / A4b / A4c | amendment 候选 |
| | A6 / A7 | 可选，最低优先级 |
| B 终态物理 | B1 / B2 | 已冻结 + 修复 |
| C 终态观察 | C1（三协议 + 误差干扰项） | 已冻结 + 升级 |
| | C2 | scan 后定 |
| E 认知 | E1 | amendment 候选（优先级在 C2 之前） |
| | E2 | scan 后定 |
| 跨头 | S-track | evaluator 即可 |
| 协议 | do(action/body/sensor/history\|state) | 采集合同 + evaluator |

**落地顺序**：

1. 纯代码先行（不动冻结语义）：B1 rank 均匀化 + 题面参考点 + public manifest 补
   body/camera/action 合同；B2 sector 证书；S-track evaluator；统计协议。
2. Amendment 一批过：A4 家族 + E1 + C1-sensor-contrast + C2（含 scan 门条件）+
   条件路由表新增行 + A3 open decision 裁决。
3. 一次 GPU replay 攒齐全部采集：可见 support 持久化（B 证书的前提，depth 目前未
   持久化）+ body paired core + sensor cell 对 + C2/E2 feasibility scan。

**一句话总结**：EgoConseq 用三个头读一次执行的后果（过程 / 终态物理 / 终态观察），
用第四个头读模型对自己知识边界的认知，用四个干预算子证明模型用的是几何而不是先验，
用 S-track 证明它能自主维护一个自洽的后果状态——所有答案由初始可见空间机械认证，
所有"不可判定"同样由几何证书裁定。它是具身执行前的后果验证器，测的是三件事：
**你知道你的身体会发生什么，你知道你会看到什么，你知道你不知道什么。**
