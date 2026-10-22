# benchmark 评测

从完整模型回答评分，所有模型共用冻结的提取、语义判断和指标实现。
题库统一在 `data/benchmark`，seen / unseen 分别包含 5,000 / 2,000 道主集题和 300 / 150 对半径诊断题。
题库和评分协议的哈希保存在发布清单与结果中。

## 正式评测前

1. 固定 release、题集、输入消融、权重版本和生成设置，核对 SFT 权重的训练数据来源。
2. 做小规模真实推理，检查图片顺序、米/厘米、空回答与截断；保留原始回答，不按得分重试。
3. 运行评分回归测试。
4. 主集（seen + unseen）和半径诊断分别报告，seen 与 unseen 分别评分后合并；诊断只有完整配置，不计入主集总分，也不做参数消融。

## 运行

在项目根目录执行。以下回答路径是完整 seen 主集推理后的示例：

```bash
bash data/benchmark/evaluation/run_eval.sh --benchmark-root data/benchmark --sets seen --responses data/benchmark/inference/responses/qwen3vl-4b_seen_answer_only_response.json --output-dir data/benchmark/evaluation/runs/seen --gpus 1
```

unseen 使用同一根目录和 `--sets unseen`；诊断集用 `seen_radius_pairs` 或 `unseen_radius_pairs`。
回答文件名以推理实际输出为准。默认根目录是 `data/benchmark`；不指定回答时只在该根目录的 `inference/`
下找回答，并按题集分组评分。独立运行目录可显式传 `--responses`。

同一题集的多个模型可一次传给 `--responses`，用 `--gpus 0,1,2,3` 分配到四个独立裁判，
共享类别判断表。单卡和多卡走同一个预检、续跑和发布入口；没有语义请求就不启动裁判。
不指定回答时，入口自动按题集处理已有的正式回答并汇总；评测配置相同的已有结果直接复用：

```bash
bash data/benchmark/evaluation/run_eval.sh --gpus 0,1,2,3
```

也可以单独合并四份汇总。`table.py` 不覆盖已有输出，重建时先写到新目录，核对后再替换 `results/`：

```bash
python -B data/benchmark/evaluation/table.py --results data/benchmark/evaluation/runs/{seen,seen_radius_pairs,unseen,unseen_radius_pairs}/summary --output-dir data/benchmark/evaluation/results_rebuilt
```

合并先按模型和输入版本收集不重叠的题集，再比较不同模型。模型权重、生成设置或评分协议
不一致会报错；每个模型保留各次 response 的路径、哈希、release 和裁判配置。
文件名是运行标识，`model_id` 是模型标识，由文件名去掉题集后缀得到，无需修改原始 JSON。
表格中的模型名取自 checkpoint 目录名，不从名字推测训练方式或模型大小。

相同命令可续跑；输入、代码或裁判版本改变时使用新输出目录。退出码 0 表示完成，1 表示输入或结果不一致。
`--background` 将日志写入输出目录的 `eval.log`。

## 评分口径

| 任务 | 评分 |
|---|---|
| Collision Occurrence（是否碰撞） | 碰撞/不碰撞准确率，另列碰撞召回率 |
| Collision Action Grounding（首次碰撞动作） | 完整动作列表中的命令编号，包含转向命令；前进动作的序数按题面解析 |
| Collision Object Category（碰撞物体类别） | 预测与 GT 同义、为其子类或其粗粒度上位类（如 chair 对 straight chair）得分；furniture、structure 等泛称、部件关系和关联物体不得分 |
| Target Distance（中途、终点距离） | 两个任务分别计算 VSI-Bench 的平均相对准确率 MRA：阈值 θ = 0.50, 0.55, …, 0.95，每题得分为满足 \|预测−GT\|/GT ≤ 1−θ 的阈值比例 |
| Target Direction（中途、终点方向） | 两个任务分别计算三轴联合准确率，另报单轴准确率 |
| Future-View Selection（未来画面选择） | 明确选中 A/B/C/D；数字或序数不转换成选项字母 |

距离先统一换算成米；只给数字时按题面单位解释。0.01 m 和 1 cm 的输出精度相同。
方向接受 Rear / Behind 等公共答案规范中的同义表达。

**主集 Avg. 等权平均八个任务**：是否碰撞、首次碰撞动作、碰撞物体类别、中途距离、终点距离、
中途方向、终点方向、未来画面。表格虽按六个名称分组，中途与终点仍分别占总分的八分之一。
All 合并 seen / unseen 的逐题结果后计算，不直接平均已舍入的分数。

具体公式：每题得分 `s ∈ [0, 1]`，距离题为 MRA，其余任务答对 1、答错 0；
任务分数 = 该任务全部题目得分的平均；`Avg = 八个任务分数之和 / 8`。
方向联合准确率要求三个轴全部正确，明确未知的轴不能补默认方向。

各任务的随机机会水平不同，因此 Avg. 是固定权重的诊断汇总，不是统一难度或随机水平校正后的
能力分数。比较模型时应同时看任务分项、seen/unseen 和半径诊断，不能只看一个总分。

半径诊断另报单题准确率和整对准确率（两题均正确），不参与主集 Avg.。
米/厘米分项是各单位题目的平均每题得分（距离题为 MRA）；它不是按单位重算的八任务 Avg.。

## 提取与校验

整批预检核对冻结 QA 哈希、唯一且完整的题目 ID、任务、题集、GT 和回答来源。
明确短答案先用规则提取；复杂回答交给盲提取器，不提供 GT，也不让它补出答案。
物体类别：不超过四个词、不含虚词（or、vs、at、of、maybe、I 等固定小词表）的名词短语按原文作为类别，
如 door frame、metallic gray table；含虚词、多行或成句的回答交给盲提取器。
动作、距离、方向和选项由程序比较；物体类别由固定裁判判断语义关系。
未选定、无效或不完整答案记 0，仍留在分母。裁判执行错误阻止完成，不作为模型答错。

裁判配置以 [extract.py](extract.py) 的 `JUDGE` / `GENERATION` 为准：Qwen3-32B-AWQ，
固定 revision，关闭 thinking、temperature=0、seed=0。保存裁判原始输出、配置、代码指纹和回答哈希。
不同模型共用评分配置，不按模型或题目单独改评分规则。

正常结束的短答案允许一个来自聊天模板预填的 `</answer>` 结束标签，完整原文仍保留。
这不适用于截断回答或未作出选择的长文本。动作编号必须存在于题面列表中；选中实际存在的
转向命令属于答错，越界编号属于无效答案，两者都记零。类别完全相同或判断表命中时不调用模型。
所有题集共用输出目录下的一张类别判断表 `category_cache.json`，同一对类别始终使用同一个判断；
该表绑定裁判配置和类别提示词，续跑和重评直接复用。
裁判启动失败直接停止，不能变成逐题反复加载模型；请求错误的有限重试仅处理执行或格式失败。

每个题集的结果目录包含 `protocol.json`、逐模型 `results/`（指标、逐题得分和提取明细）及 `summary/`。
完整结果可用 [table.py](table.py) 合并：`--results` 指定结果目录，`--output-dir` 指定新汇总目录。

## 实现参考

- [MMBench / VLMEvalKit 官方评测](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/dataset/utils/multiple_choice.py)：
  参考规则预提取、困难答案再调用裁判、并发与结果缓存。本实现不会在提取失败时随机补答案。
- [VSI-Bench 官方评测](https://github.com/vision-x-nyu/thinking-in-space/blob/main/lmms_eval/tasks/vsibench/utils.py)：
  参考按任务聚合再汇总的组织方式；距离 MRA 与其实现一致（阈值 0.50:0.95:0.05，含边界，无法解析记 0）。
