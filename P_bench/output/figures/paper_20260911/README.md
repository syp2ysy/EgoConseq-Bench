# Editable benchmark figures

All six numbered figures are drawn with Python / Matplotlib from the current benchmark. SVG text stays text, heatmap cells and chart marks stay vector shapes, and PDF fonts are embedded. PNG files are previews. No generative image model is used for these figures.

Run from the project root:

```bash
python visualization/plot_benchmark_figures.py
```

An alternative output folder or resolution can be selected with `--output DIR --dpi 300`. Runtime dependencies: Python, NumPy, Matplotlib. Edit `TASKS`, `GROUPS`, `TASK_COLORS`, `SPLIT_COLORS`, and `rcParams` near the top of the script to change names, ordering, colors, and typography; individual figure functions control layout.

## Research question and classification

Navigation decisions precede the observations produced by executing an action. A foundation model therefore needs to relate current visual evidence to a specified robot configuration and motion to predict spatial states that have not yet been observed through execution. This benchmark directly queries body–scene contact, target-relative distance and direction at intermediate and final poses, and the final camera view through candidate selection. These outputs provide verifiable measurements of spatial prediction under ego-motion; they do not establish a particular internal simulation mechanism or closed-loop navigation performance.

The taxonomy has two distinct axes: **prediction object** (contact, target relations, visual observation) and **query scope** (entire motion, an intermediate position, final position). Environment, embodiment and motion are joint input conditions, not disjoint task categories. In particular, A4 belongs to the target-relation family despite its A prefix. The displayed taxonomy describes tasks, not a proven hierarchy of cognitive difficulty.

| Prediction family | Full task name | Scope and output |
|---|---|---|
| Body–scene contact | Collision occurrence (A1) | Entire motion; Collision / No collision |
| Body–scene contact | First-collision action (A2) | Collision is given; index in the full listed action sequence |
| Body–scene contact | First-contact object category (A3) | Collision is given; category of the first contacted object |
| Target-relative spatial relations | Intermediate target direction (A4) | Camera-frame horizontal and vertical direction at the specified intermediate position |
| Target-relative spatial relations | Final target distance (B1) | 3D distance from camera optical center to the fixed marked point after all actions |
| Target-relative spatial relations | Final target direction (B2) | Camera-frame horizontal and vertical direction after all actions |
| Target-relative spatial relations | Intermediate target distance (B3) | 3D camera-to-point distance at the specified intermediate position |
| Future-view discrimination | Future-view selection (C1) | One of four alternative final views; not image generation |

Intermediate positions are defined after all previous actions and 25%, 50%, or 75% of the distance in one specified Forward action. Target-relation and future-view sequences are collision-free. The complete given sequence length is distinct from the prefix needed to reach an intermediate query.

## Figure files and captions

Each name below has `.svg`, `.pdf`, and `.png` versions.

1. `01_task_taxonomy`: **Task taxonomy for spatial prediction under ego-motion.** Current egocentric evidence, robot configuration, and a given motion sequence jointly define the prediction problem. Eight tasks query contact, target-relative spatial relations, and future-view discrimination. Questions are shortened explanatory paraphrases, not replacements for the benchmark prompts. Candidate rectangles are schematic alternatives, not benchmark images.
2. `02_task_distribution`: **Question composition and evaluation splits.** The inner ring groups questions by prediction family; the outer ring contains all eight named tasks. All angular proportions and displayed percentages use the total number of benchmark QA items as denominator. Horizontal bars report absolute task counts separately for Seen and Unseen.
3. `03_configuration_and_motion`: **Robot configuration and motion coverage.** Configuration heatmaps use the same count scale; each cell counts QA items for a body-radius / camera-height / horizontal-FOV combination. Bars show within-split question percentages, with raw counts printed above them. Source bars count QA items, not unique scenes. Eighteen configurations denote parameter combinations, not eighteen different robot platforms or paired counterfactual scenes.
4. `04_distance_and_query_progress`: **Metric and intermediate-query coverage.** Distance histograms use the published distance GT, common 1 m bin edges, and normalization within each task and split; dashed lines mark medians. Progress bars count intermediate queries at fractions of one Forward action. Fractions do not denote total trajectory time or length.
5. `05_source_and_answer_coverage`: **Source coverage and answer-label balance.** Heatmaps contain source-by-task QA counts on a shared scale; zeros indicate no samples. Collision and candidate-answer bars show within-task, within-split proportions; labels show absolute counts. These are GT distributions, not model accuracies or outputs. The A–D axis denotes candidate choices for the fully named Future-view selection task.
6. `06_target_direction_coverage`: **Camera-relative direction labels.** Each task and split is decomposed into eight horizontal directions and three vertical categories. Cells show counts and within-task, within-split percentages; colors use a shared percentage scale. The 24 labels do not cover equal solid angles. Metadata uses `rear`; the synonymous published answer `behind` is canonicalized for validation.

## Relation to prior presentation conventions

- [VSI-Bench / Thinking in Space, Fig. 5](https://arxiv.org/html/2412.14171v2): prediction/ability families and named leaf tasks in a hierarchical composition chart; input-complexity distributions are shown separately.
- [MMSI-Bench, taxonomy and category distribution](https://arxiv.org/html/2505.23764v1): classification by spatial entities and relations, with task names and a shared QA denominator for hierarchical proportions. We follow the semantic organization and explicit denominator, without importing unrelated task categories.
- [ActionEQA, final TMLR paper](https://limanling.github.io/uploads/paper/ActionEQA.pdf): separate task axes, action magnitude distributions, and answer-position statistics. Its state-prediction setting already tests action-conditioned future-state selection, so future prediction alone is not asserted as novel here.
- [ENACT, final ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/f841d4e62cf309aec2bfe9a459e852b4-Paper-Conference.pdf): sequence-length composition and action-conditioned future-observation ordering. Full sequence length in our charts counts Forward/Turn actions; it is not ENACT's keyframe length.
- [EgoDyn-Bench, Fig. 2](https://arxiv.org/html/2604.22851v1): conditional positive-label coverage is visualized with grouped bars. Our answer-balance bars likewise describe label distributions, not model accuracy.

The three prediction families are a taxonomy for this benchmark, informed by those presentation conventions. They are not attributed to a prior paper as an existing standard.

## Data and checks

Authoritative input: `data/benchmark/benchmark/{seen,unseen}/QA.json`, joined by QA ID to `data/benchmark/metadata/{seen,unseen}/record_index.json`. Only IDs in the public QA files are counted. SFT examples, ablated copies, inference responses, and evaluation results are excluded.

`figure_data.json` stores input SHA-256 fingerprints, exact counts, configuration cells, source-by-task coverage, distance GT values, progress counts, and direction matrices. `task_counts.csv` contains the eight named tasks and their split counts. Both are regenerated by the script.

The script checks metadata/QA fingerprint agreement, unique IDs, matching task/source labels, valid plotted condition ranges, count totals, histogram coverage, and direction metadata agreement with published GT. Current inputs contain 4,999 Seen plus 2,000 Unseen QA items across 170 source-qualified scenes. No benchmark prompts, GT, images, or model evaluations are modified.
