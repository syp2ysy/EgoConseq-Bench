# 答案提取与评分

除 A3 外，流程为：`response JSON 的 answer → 正则或 LLM 提取 → 统一规则评分 → 汇总准确率`。
能直接识别的短答案用正则；其余回答交给 LLM 提取，包括有多个数值的长距离回答。
明确的目标—相机关系句也直接解析；否定、多个时刻或其他参照物不会仅凭方向关键词匹配。
这些任务的 LLM 接收实际问题、查询时刻、题中方向/距离定义、动作总数和原始回答，不接收 GT、不生成分数。
提取输入去掉身体参数、具体动作列表和类别表；保留动作总数用于核对回答的运动时刻。
A3 的每条原始回答与 GT 一起交给同一个 LLM，直接判断类别是否语义一致，再汇总总体准确率。

| 任务 | 提取内容 | 评分规则 |
|---|---|---|
| A1 | collision / no_collision | 与 GT 一致 |
| A2 | Action/Step 编号，或第几个 Forward | 将 Forward 排名映射到包含转向的完整动作列表，再比较编号；不修正明确编号 |
| A3 | LLM 判断首次接触类别是否与 GT 语义一致 | 正确题数 / A3 总题数；不按物体类别分组或平均 |
| A4 / B2 | 水平方向＋垂直方向 | 分别比较前后、左右、上下轴；主准确率要求三轴全对，另保留逐轴分 |
| B1 / B3 | 所查询时刻的最终距离，保留数值、正负号、单位 | 换算为米后分别计算误差 ≤0.25 m 和 ≤0.5 m 的准确率，包含边界 |
| C1 | 最终选择的唯一 A/B/C/D | 与 GT 选项一致 |

A3 先确认回答给出了唯一的最终类别，再判断是否同义；“沙发或椅子，无法确定”即使包含 GT 也计错。
按室内场景中的通常含义接受同义词和常用别称，例如 `couch = sofa`、`pillar = column`、
`stairs = staircase`；不对这些常见同义词强加专业分类差异。允许不改变类别的颜色/材质修饰，例如 `wooden wall = wall`。
相关物体或部分与整体不能当作同义词，例如 `floor ≠ wall`、`mattress ≠ bed`；
对确实存在粗细粒度差别的类别，要求回答足够具体：`chair` 不能据此认定为 `swivel chair` 或 `armchair`，
`table` 不能认定为 `breakfast table`，`cabinet` 不能认定为 `base cabinet`，`lamp` 不能认定为 `floor lamp`。
若 GT 本身使用泛类，则明确的子类可以满足它，例如回答 `swivel chair`、GT 为 `chair` 可判对。
只判断最终明确声称的首次接触物体；不能因回答中提及 GT、否定 GT 或称其为后续接触物体而给分。
空回答、拒答或未确定的多个选项计错。裁判引用原回答并返回布尔 `correct`，
程序按 1/0 汇总；不再使用固定别名表评分。Seen 和 Unseen 分别统计各自全部 A3 题目。

距离用 Decimal 比较，支持 m/cm/mm；无单位按米处理。负距离、非有限值、无确定答案均不给分。
不把动作长度、机体尺寸、相机高度或中间计算结果当作目标距离。区间没有明确选定数值时，
提取为 null，不取中点。±0.5 m 是附加指标；总体 macro 仍采用距离 ±0.25 m 的准确率。

水平方向为 front/rear/left/right/front-left/front-right/rear-left/rear-right；
垂直方向为 above/level/below。明确水平朝向但未说明高度时按 level；明确表示高度未知则为 null。
只有 Below 时水平为 null。Front 对应前方和左右居中；Left 对应前后中性和左方。
未知轴不给分。提取时使用所查询时刻的“目标相对相机”关系，不能混入动作描述或其他参照物。
`forward and left` 是 `front-left`，不是 `left and above`；相机在目标上方表示目标在相机下方。
仅说明相对地面或桌面的高度，不能据此补出相对相机的上下关系。
同样，目标相对地面中心、窗户等物体的左右关系不等于目标相对相机的左右关系；
background 不等于 rear，房间中心不等于视野正前方。不能把不同运动时刻的分量拼成答案。
方向分量明确冲突时该分量为 null；只有 `Forward 0.5 m` 这样的动作指令不提供目标方向。
North/east 等绝对方位需要回答明确给出所查询时刻的相机罗盘朝向；缺少参照时不能直接映射为 front/right。
例如 `east and above` 提取为水平 null、垂直 above。复杂角度句交给 LLM 保留角度及参照，再由 Python 划分方向。
明确给出相对正前方的角度时，复用题目生成时的方向划分规则：四个正方向各覆盖 ±7.5°，包含边界。
例如 `7.5 degrees left of forward` 仍为 front；相机坐标系问题中的 `left 45 degrees`
按距正前方左侧 45° 解释为 front-left，不能丢掉角度仅取 left。
只有 `194.66 degrees` 或 `Horizontal: 102.5°` 时不猜测角度零点。
无参照角度同时给出 above/below 时，仅保留其垂直分量。

LLM 统一先引用原回答中的完整结论 `evidence`，再用一句 `reason` 解释其含义，最后返回：
- A3：布尔 `correct`；
- 其他任务：`answer`；方向题是 `{"horizontal": ..., "vertical": ...}`，未知分量为 null。

程序核对引用确实出现在原回答中；能直接解析的引用继续使用正则规范化，防止裁判改动数值或补出方向。
没有答案、拒答或无法消解的多个答案返回 null，计入分母并计错。
格式错误、引用不存在或裁判输出截断时重试一次；仍失败记 `extraction_error`，保留原文和错误供续跑。
裁判故障不计作被测模型答错，也不从分母中偷偷排除。任何题目未评分时，对应任务指标保持 null。
这与模型输出无效答案得 0 分不同。只有全量评分完成后，才生成最终数字表格。

```bash
bash data/benchmark/evaluation/run_eval.sh \
  --responses data/benchmark/inference/responses/cosmos3-edge_thinking_off_response.json
```

默认本地 Qwen3-8B、开启 thinking；固定 temperature=0.6、top_p=0.95、top_k=20、seed=0，
batch 8，最多 4,096 个输出 token。使用采样而非 thinking 模式的贪心解码。
只解析其思考结束后的最终 JSON，保留完整提取/裁判原文供核查；这不改变被测模型的推理设置。
`--rules-only` 只执行无需 LLM 的部分，A3 全部保持待评分；同一命令去掉该参数即可续跑。
已有 API 使用 `--extractor-model MODEL --base-url URL`，密钥变量为 `INFER_API_KEY`。

每个模型保存两份文件：`full__extract-Qwen3-8B.json` 包含原始回答、提取结果、提取方式与得分；
`full__extract-Qwen3-8B_scores.json` 包含分数汇总。距离列为 `accuracy_at_0.25m` 和 `accuracy_at_0.5m`。
A3 明细保存 `judge_output`、`scoring_method: "llm"`、`correct` 和 `score`；汇总读取 `tasks.A3.accuracy`。
尚未完成的任务准确率保持 null；完成后才计算完整指标。`--check-only` 只检查，未完成时退出码为 3。
当前协议为 `extract_and_judge_v7`。续跑校验完整题目集合、QA 哈希、原始 response 哈希、
提取器权重、生成参数和评分代码哈希；本地 API 端口变化不影响续跑。
协议发生变化后须对全部模型重新评分，不复用旧协议的 LLM 判断；相同协议可断点续跑。
裁判输入不包含被测模型名称。不同模型共用一份规则，不设模型专用解析分支。
所有原始回答均保留 `response_finish_reason`，任务汇总中的 `truncated_responses` 统计截断数量。
截断本身不直接决定得分：有明确最终答案就按规则评分；没有最终答案时不把中间推理补成答案。

五个小模型的完整推理与评分可通过 `../inference/run.sh small --cached --serve --evaluate` 顺序运行。
队列使用同一个已缓存 Qwen3-8B checkpoint，API batch/workers 为 16，最多 4,096 个输出 token，
上下文长度 16,384。队列与各模型日志位于 `../inference/logs/`，状态见 `../inference/pipeline_status.json`。
正式评分只有本目录的 `eval.py` 一个入口；旧的 `scripts/eval_benchmark.py` 已移除。

五模型队列全量评分完成后自动调用 `table.py`，生成 `results/five_models_seen_unseen.tex`。
也可手动运行 `python data/benchmark/evaluation/table.py`。
表格按 Seen / Unseen 两大列、各八个子任务组织，数字不加粗。
生成前核对全部模型评分完整、每项任务题数相同，且输入版本、裁判权重、生成参数与评分协议一致；
不符合条件时直接停止，不输出带 `--` 的最终表格。
