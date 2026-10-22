# Benchmark 统计与 QA 审阅

这是发布数据的只读视图，独立于采集、选题和评测。Python 读取发布目录中的
`release.json`、`metadata/selection.parquet`、`report.json`、公开 QA 及其引用的 meta；
静态 HTML/CSS/JS 按当前筛选计算图表，可以从图表直接进入对应的完整 QA。

## 重算

在项目根目录、具备项目 Python 依赖的环境中运行：

```bash
python -m scripts.review_page data/benchmark \
  --out data/analysis/benchmark_review
```

后续更新 benchmark 后，用同一命令重新生成；一个根目录自动包含 seen 和 unseen。
元数据位置由 `release.json` 中的分区声明指定，不读取当前可变的选题 spec，不启动模拟器。
只计算入选题用到的原始图和选项图指标；同一图片在一次生成中只计算一次。

页面入口：`data/analysis/benchmark_review/index.html#stats`。
现有本地服务对应 `http://localhost:8765/analysis/benchmark_review/index.html#stats`。
如需自行启动，可运行 `python -m http.server 8765 --directory data`。

输出为 `items.js`（逐题字段）、`stats.json`（发布摘要、来源与统计口径）和静态界面。
页面可导出当前筛选的逐题统计 JSON；每张图均说明分母和参考方法。
布局调整后可用 `--layout-only --out ...` 复用已生成的统计缓存。

## 口径

- 沿用三个后果域和八项任务名称；任务代号只作为内部关联键。
- 主 benchmark 与半径诊断集分开统计。参数消融版本不重复计数；一对诊断题是两题、一个观测。
- 题数、观测数、未打点初始图数、实际输入图数分别统计。图片按发布的内容哈希识别。
- 所有物理长度统一为米；题面中的米/厘米表达单独计数。
- 区分公开命令数、累计运动量、查询动作前缀和程序 ID 集中程度。
- 分档柱图显示区间样本数，不是概率密度。不同宽度区间不据柱高推断密度。
- 多数答案占比是描述性参考；距离的初始值基线使用 GT 信息，只是诊断。没有虚构无图模型或人工成绩。
- 图片质量、深度一致和路径可见性只展示证据，不修改过滤规则，也不把贴脸/低纹理直接判为异常。
- 构建供给表使用历史完整报告，可能包含训练候选且格子间共享观测；不与当前筛选分母混用。
- 浏览器人工审阅记录使用稳定的分区标识；本次整理保留原标识，移动目录不会清除记录。

图表参考 VSI-Bench、MMSI-Bench、RoboSpatial、ActionEQA、GQA、CLEVR、MMBench、
Video-MME、ECBench、OpenEQA、SPAR 的原论文或官方页面。
具体图号、链接和适配关系维护在 `scripts/review_stats.py`，并显示在页面中。
项目专有的动作、半径、查询前缀等字段是对相应统计方法的适配，不冒充原论文指标。
