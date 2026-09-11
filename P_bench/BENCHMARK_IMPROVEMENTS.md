# ABC1 records 与 seen-5000

更新：2026-09-06。

## records 与采集边界

当前库：`data/metadata/train/seen_updates/current`，schema `abc1.record.v3`。
B1K 已停止扩采并收尾：11,750 条；GS 24,569；R2R 22,238；总计 58,557。
停止不是凑到原 12,000 目标；仅计入实际已完整落盘的数据。

- records refine 不修改 pose、现有半径、高度、FOV、分辨率或初始图。
- scene-first：每卡一次加载一个场景，处理完该场景全部指派 records 才换场景。
- 新场景使用各数据集原有场景划分和原生 pose 采样，写同样的 compact schema。
- 初始 Turn 为 ±15°/±30°；中心线 FOV 与真实半径的 corridor/物理碰撞分别判定。
- 每个 case 保存 actions、碰撞/终点、来源 protocol/variant 和可编译 task_outputs。
  表面点保存 point_id、真实 mask/depth 像素及 world_xyz_m；不使用 bbox 中心冒充 GT。
- A4 用冻结的 Forward 内部 checkpoint；B2 用包含末步 Turn 的最终 pose。
  B1 距离变化至少 max(0.30 m, 初始距离的 10%)，终点距离至少 0.05 m。
- A4/B1/B2 和 C1 全部安全且执行完整；A1 支持 safe/collision，平衡放在选题端。
  A2/A3 当前仍是首次碰撞动作/接触类别任务。GS 不生成 A3/B1。
- C1 只在 record 级 c1_families，query 和三个候选为完整安全 family，保留像素/PNG 去重。
- 不要求单条 record 集齐所有任务，也不在 records 中保存 DINO embedding。

## seen-5000 选取

固定 dataset/task 总数见 `post_QA/specs/seen_5000_v1.json`。
5,000 条不同 record，各贡献一道 QA；private/record_index.json 是 SFT 整条 record
排除的权威输入。旧 6,470/7,400/8,400 常量不再定义新发布物。

1. 结构化单遍扫描：按 record/任务/起手/答案/长度保留轻量 case/point 引用，
   再构造少量候选；不先展开全部数百万 QA。
2. 场景与动作/答案轮换产生首批候选；仅候选初始图执行质量检查与 DINO。
   黑区、连通黑洞、空白/模糊阈值和模型输入尺寸都记录在最终 report。
3. DINOv2-S/14 直接加载预训练权重，336×252 完整视野输入，归一化特征 cosine。
   分块计算直接近邻，不存 N×N 矩阵、不做传递聚类、不落盘 embedding。
4. 全局 record-exclusive 匹配同时排除图像近邻。缺额才扩大候选池，已编码图复用。
5. 普通任务/C1 优先均衡 L1–L6；A2 仅 L3–L6，L3 Turn-first=0，
   每个 length/start 内 ordinal/rank 平衡。A2/C1 长度配额从合格供给分配，
   不冻结不可达长序列小格。各 dataset/task Turn-first 至少 30%。
6. 只为最终题目物化标记图和 hardlink 原图；不重写源 records。
   C1 终点图检查质量，内部选项不套用跨题 DINO 阈值。
7. 输出 QA、引用图片、private index、一份 report 和内嵌数据的 HTML。
   HTML 提供任务/数据集/场景/长度/物理配置过滤、GT 和最相近图对。

A2 的最长 Forward baseline 和按起手类型条件化的随机 Forward baseline 会记录在 report。
交替序列的碰撞 index 奇偶先验是结构约束，不宣称已经消除。

## A4/B2 的 24 类方向均衡

方向仍按 8 个水平方向 × 上/同高/下，共 24 类；不合并正前、前左和前右，
不修改几何分界或 GT。候选预筛保留每个方向的 case/point，再在相同
dataset/task/起手/长度内替换，使 A4、B2 各自的 700 题尽量均衡到每类 29–30 题。
同一 record 仍只贡献一道 QA；新换入的初始图须通过同一质量与 DINO 近邻限制。
若供给不足，report 如实记录覆盖和各类数量，不凭空补标签。

`rebalance-directions --benchmark-root data/benchmark/seen_5000` 只更新 A4/B2，
其他 3,600 题不变，源 records 不变；同步更新 QA、SFT 排除索引、report 和 HTML。
HTML Overview 分别展示两张完整 8×3 计数表，包含零计数格；点击任一方向可查看
对应题目，Cases 页也支持精确方向筛选。正常全量构建复用同一方向均衡逻辑。
DINO 仅编码现有选中图与按需补选图，特征不落盘；发布后自动删除临时目录。
本次发布替换 827 题：A4 每类 28–30 题，B2 每类 29–30 题，均覆盖 24/24；
保留原 dataset/task/起手/长度数量，最终最大图像 cosine 为 0.899826（阈值 0.90）。

## 标点与题面刷新

A4/B1/B2 每题只画一个无编号标点：红色点心半径 6 px、白环到 8 px、
黑色外环到 10 px，圆心仍为原始 `pixel_xy_px`。公共题面用 `the marked point`，
不再附带物体类别；内部 point_id、类别、世界坐标仍保留在 records/private index。
A4 明确查询指定 Forward 距离完成 25%/50%/75% 时；B1/B2 查询全部动作执行后，
包括最后的 Turn。相机参考系、方向划分、距离评分和 GT 不变。

`refresh-presentation --benchmark-root data/benchmark/seen_5000` 从现有索引
定向读取 2,000 条源记录，只重画标图、更新题面和摘要、重建 HTML。不重新筛选、
不运行模拟器或 DINO，也不改 SFT 排除集合。未改文件临时 hardlink；切换失败恢复
原目录，成功后删除旧展示文件和临时目录。report 中记录 `surface-dot-v2`。
Benchmark/SFT 继续共用绘图与模板；三类模板各保留三条，删除字体、点编号及重复
Forward ordinal 的题面格式化代码，不新增 records gate。

## 代码职责

- `pipeline/`：新场景采集、固定 pose refine、compact records 与任务事实。
- `post_QA/seen_build/selection.py`：结构化候选与唯一 record 匹配。
- `images.py`：初始图质量、DINO 特征和直接余弦近邻，全部内存态。
- `visual_selection.py`：图像约束、动作长度分配和场景/答案多样性。
- `direction_balance.py`：24 类方向统计与同配额替换，共用内存中的图像特征。
- `release.py`：records 物化；或独立地从现有 records 编译 benchmark。
- `sft.py`：排除 benchmark records 后导出每条记录的多个 QA。
- `visualization/benchmark_browser.py/.html`：一份可直接打开的 HTML，不再复制 cases.json。

旧“重写整库后再 build/publish benchmark”的平行入口、无消费者的 B1 分箱、
过期 6470 规格与固定常量测试已移除。新场景 background/capacity/source manifest、
实际使用的 B1K legacy observation profile 仍保留，不能按 raise 数量误删。

## 生命周期与清理

正式 records、最终 QA/图片、SFT 排除索引不可当缓存删除。构建失败只清理自己的
空目录/临时标记图，不触碰源库。DINO 权重是复用资产，不是本轮垃圾。
完成迁移和发布验收后，删除被替代 records 版本、已结束 worker 的临时目录与
旧可重建语义缓存；先确认当前图像为独立文件/hardlink，不依赖旧版 symlink。
清理不运行新的 oracle，不改 record 语义，不生成多套快照或审计副本。
