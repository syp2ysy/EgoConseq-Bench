# Paper 页动机与设计理由增补 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `scripts/egoconseq_paper.html` 上新增动机章节、13 项对比勾叉表、逐节设计理由折叠框、do(body) 翻转示意面板、9 张 QA 设计卡 + GT 导出流水线，面向导师/合作者初次讲解。

**Architecture:** 单文件 standalone HTML（内联 CSS，无外部 script/样式表——既有测试锁死此约束）。所有新内容为纯静态节段插入，不改动现有 8 节正文，只在其中追加折叠框。TDD 载体是 `tests/test_pl_paper_page.py` 的结构契约测试（stdlib HTMLParser）。

**Tech Stack:** 纯 HTML/CSS；pytest（`PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python`）。

## Global Constraints

- Spec：`docs/superpowers/specs/2026-07-30-paper-page-motivation-design.md`；事实与措辞唯一来源是 `docs/archive/2026-07-27-benchmark-design-freeze.md` §0/§1/§2/§3/§6。
- 措辞红线：页面任何位置不得出现"首个"、"没有先例"、"the first benchmark"；差异表述用 "operationalize the conjunction" / "did not find the same controlled combination"。
- 未跑实验（盲基线、gain/leakage、human gate 数字）只讲机制，不出现任何数字。
- 只修改两个文件：`scripts/egoconseq_paper.html`、`tests/test_pl_paper_page.py`。不触碰 pipeline 代码、不动 8770/8773 服务、不改契约文档。
- 保持 standalone：不新增 `<script src>` / `<link rel="stylesheet">`（既有测试 `test_paper_page_is_standalone_responsive_printable_and_accessible` 会失败）。
- 现有 8 节的既有正文措辞不改动；新增节编号用 Section 0 / 0.5，现有锚点 id 不变。
- 工作目录：主仓 `/home/zhangshan/syp/myvln/P_bench`（两条采集均已完成，无进程风险）。
- 每个 Task 结束跑目标测试文件 + 提交一次；最后 Task 跑全量 pytest。

---

### Task 1: Section 0 · Why This Benchmark（动机三层递进 + 克制条款）

**Files:**
- Modify: `scripts/egoconseq_paper.html`（`</header>` 之后、`<section id="question">` 之前插入；`<style>` 末尾追加 CSS）
- Test: `tests/test_pl_paper_page.py`

**Interfaces:**
- Produces: 节 id `why`；CSS 类 `.whycol`、`.restraint`、`.litref`（Task 2/3 不依赖，独立）。

- [ ] **Step 1: Write the failing test**（追加到 test 文件末尾）

```python
def test_paper_page_states_motivation_and_restraint():
    html, parser, text = _page()
    ids = {attrs["id"] for _tag, attrs in parser.tags if "id" in attrs}
    assert "why" in ids
    for marker in (
        "应用层", "科学层", "方法层",
        "body-conditioned", "evidence-bounded",
        "发布契约而非被测能力",
        "CapNav", "RoboSpatial",
        "does not establish that a model internally represents a complete "
        "world model",
    ):
        assert marker in text
    # 措辞红线（全页生效，从本任务起持续检查）
    assert "首个" not in text
    assert "没有先例" not in text
    assert "the first benchmark" not in html.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_pl_paper_page.py::test_paper_page_states_motivation_and_restraint -v`
Expected: FAIL（`"why" in ids` 断言失败）

- [ ] **Step 3: Implement — 插入 Section 0 与 CSS**

在 `<style>` 内 `@media print` 规则**之前**追加：

```css
.whycols{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:18px 0}
.whycol{background:var(--card,#fff);border:1px solid var(--line,#e3ddd0);border-radius:10px;padding:14px 16px}
.whycol b.tier{display:block;font-size:13px;letter-spacing:.08em;color:var(--accent,#8a5a2b);margin-bottom:6px}
.litref{background:rgba(138,90,43,.06);border-left:3px solid var(--accent,#8a5a2b);padding:10px 14px;margin:14px 0;font-size:14px}
.restraint{border:1px dashed var(--line,#e3ddd0);border-radius:10px;padding:14px 16px;margin:16px 0;font-size:14px}
@media(max-width:880px){.whycols{grid-template-columns:1fr}}
```

在 `</header>` 之后插入（措辞照抄契约 §1/§0，白话句为新增讲解）：

```html
<!-- ================= 0. 动机 ================= -->
<section id="why">
  <h2><small>Section 0 · Why This Benchmark</small>为什么要做这个 benchmark：三层动机</h2>
  <p class="lede">在看"是什么"之前，先回答"为什么"。我们的动机分三层递进：
  应用上缺一个直接的前置测试，科学上有两个被系统性忽略的性质，方法上我们能把
  评测提升到可审计的因果干预层。</p>

  <div class="whycols">
    <div class="whycol"><b class="tier">应用层</b>
      VLM 正被接入具身系统充当规划"大脑"，但现有评测很少直接回答前置问题：
      它能否从第一视角预测"<b>我这个身体</b>、做<b>这个动作</b>、会发生什么"。
      白话：机器人把方向盘交给 VLM 之前，没人考过它"科目二"。</div>
    <div class="whycol"><b class="tier">科学层</b>
      物理后果有两个被系统性忽略的性质。<b>body-conditioned</b>：同一场景、
      同一动作，半径 0.2 m 的身体过得去、0.3 m 的过不去——这是具身智能区别于
      旁观者视觉的定义性性质，但常用评测把身体当常量或没有身体。
      <b>evidence-bounded</b>：物理上有确定答案，但单张图未必支持这个答案
      （拐角看不见）——我们把它处理为<b>发布契约而非被测能力</b>：预注册的
      可视支持契约筛选发布集，契约不满足的不发布，不测"模型是否知道证据不足"。</div>
    <div class="whycol"><b class="tier">方法层</b>
      我们的读出在<b>后果级</b>：GT 是可判定的二/三值物理事实，不是感知相似度，
      因此可以做真正的 do() 干预实验。许多像素级短时预测评测在"动作可控性"层
      仍主要依赖感知代理指标（Δ-LPIPS、round-trip consistency 等）；我们在同一
      层给出可审计物理真值上的精确干预。</div>
  </div>

  <div class="litref"><b>文献先例：为什么"换身体条件翻转 GT"是最强的具身论证。</b>
  CapNav（arXiv 2602.18424）让同一任务在五种 embodiment 下可行率从 1.0 掉到
  0.22；RoboSpatial（arXiv 2411.16537）让同一问题在 ego/world/object 三参照系下
  答案不同——embodiment-agnostic 的模型在这类结构上必然失败。我们做的是该论证
  的最受控版本：<b>固定单张标定图、reset-matched、连续半径、真实模拟器重
  rollout</b>——身体是公开的连续干预变量，而不是离散 profile 或参照系标签。</div>

  <div class="restraint"><b>克制条款（契约 §0 冻结，原文）：</b>
  <i>"Our evidence concerns behavioral outcome fidelity and counterfactual
  responsiveness. It does not establish that a model internally represents a
  complete world model or possesses long-horizon planning and policy-evaluation
  capabilities."</i>
  我们评估的是行为层面的 action-conditioned consequence 能力，不主张证明模型
  内部维护了完整世界模型。</div>
</section>
```

- [ ] **Step 4: Run tests to verify pass（含既有全部页面测试）**

Run: `$PY -m pytest tests/test_pl_paper_page.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/egoconseq_paper.html tests/test_pl_paper_page.py
git commit -m "feat(paper-page): add Section 0 motivation with restraint clause"
```

---

### Task 2: Section 0.5 · Related Benchmarks（13 行勾叉对比表）

**Files:**
- Modify: `scripts/egoconseq_paper.html`（Section 0 之后插入；CSS 追加）
- Test: `tests/test_pl_paper_page.py`

**Interfaces:**
- Produces: 节 id `related`；CSS 类 `.cmpwrap`、`.cmp`、单元格类 `.y`（✓）/`.p`（△）/`.n`（✗）/`.na`（—）。

- [ ] **Step 1: Write the failing test**

```python
def test_paper_page_compares_related_benchmarks_honestly():
    html, parser, text = _page()
    ids = {attrs["id"] for _tag, attrs in parser.tags if "id" in attrs}
    assert "related" in ids
    for name in (
        "VSI-Bench", "BEAR", "PhysBench", "RoboSpatial", "Physion",
        "CLEVRER", "TouchSafeBench", "CapNav", "WM-ABench",
        "World Models in Words", "PHYRE", "Lost in Aggregation",
    ):
        assert name in text
    arxiv_links = [
        attrs for tag, attrs in parser.tags
        if tag == "a" and str(attrs.get("href", "")).startswith(
            "https://arxiv.org/abs/")
    ]
    assert len(arxiv_links) >= 12
    # WM 阶梯：编号未核实到，保留行但不挂链接
    assert "编号待复核" in text
    # TouchSafeBench 采用修正后的轴描述（表征×视角，非 morphology 轴）
    assert "表征 × 视角" in text
    # 冻结差异声明原句
    assert "did not find the same controlled combination" in text
    assert "reset-matched continuous-radius consequence protocol" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_pl_paper_page.py::test_paper_page_compares_related_benchmarks_honestly -v`
Expected: FAIL（`"related" in ids`）

- [ ] **Step 3: Implement — CSS + 对比表节**

CSS（追加在 Task 1 新增块之后）：

```css
.cmpwrap{overflow-x:auto;margin:16px 0;border:1px solid var(--line,#e3ddd0);border-radius:10px}
.cmp{border-collapse:collapse;font-size:13px;min-width:980px;width:100%}
.cmp th,.cmp td{border-bottom:1px solid var(--line,#e3ddd0);padding:7px 9px;text-align:center;vertical-align:top}
.cmp th{background:rgba(138,90,43,.07);font-size:12px}
.cmp td.w{text-align:left;white-space:nowrap}
.cmp td.note{text-align:left;min-width:220px;font-size:12px;color:var(--muted,#6f6a5e)}
.cmp .y{color:#1c7c3c;font-weight:700}
.cmp .p{color:#a8730a;font-weight:700}
.cmp .n{color:#b3392e}
.cmp .na{color:var(--muted,#6f6a5e)}
.cmp tr.ours{background:rgba(28,124,60,.07);font-weight:600}
```

节 HTML（插在 `</section>`（Section 0）与 `<section id="question">` 之间）。
勾叉矩阵按下表逐格填写（✓=`<td class="y">✓</td>`，△=`<td class="p">△</td>`，
✗=`<td class="n">✗</td>`，—=`<td class="na">—</td>`）；备注列内容照抄契约 §3
"关键设计装置"（TouchSafeBench 一行用修正描述）：

| 行（链接） | 单图标定 ego | 身体=连续干预变量 | 度量动作程序 | 重rollout GT | reset-matched do() | 配对联合计分 | 可视支持契约 | 备注（关键设计装置） |
|---|---|---|---|---|---|---|---|---|
| VSI-Bench `2412.14171` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 视频空间智能；3 类 8 任务；盲答验证 + Frequency baseline |
| BEAR `2510.08759` | △ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 6 类 14 原子技能；感知→规划层级；图像非标定 |
| PhysBench `2501.16411` | △ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 属性/关系/场景/动态四域 19 子类；image-video-text 混合 |
| RoboSpatial `2411.16537` | △ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ego/world/object 三参照系标注；我们借鉴其参考系纪律 |
| Physion `2106.08261` | ✗ | ✗ | ✗ | △ | ✗ | ✗ | ✗ | 8 场景统一二值接触读出；人机同刺激 forced choice 先例 |
| CLEVRER `1910.01442` | ✗ | ✗ | ✗ | △ | ✗ | ✗ | ✗ | 四读出分离；其 counterfactual 移走物体，我们的改变自己 |
| TouchSafeBench `2605.31196` | ✗ | ✗ | ✗ | △ | △ | ✗ | ✗ | 表征 × 视角受控消融（morphology 固定）；碰撞分类非度量 rollout |
| CapNav `2602.18424` | ✗ | △ | ✗ | ✗ | △ | ✗ | ✗ | 5 种离散 embodiment profile；路线级可行性翻转 |
| WM-ABench `2506.21876` | ✗ | ✗ | ✗ | △ | △ | ✗ | ✗ | 23 原子维度 × 6 模拟器；受控反事实；广义 WM 维度 |
| World Models in Words `2605.29585` | △ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 单图 + typed 语言 trace 混合验证；我们的终局由模拟器机械导出 |
| PHYRE `1908.05656` | ✗ | ✗ | △ | △ | ✗ | ✗ | ✗ | 2D 物理解谜；跨谜题泛化的 split 纪律先例 |
| WM 阶梯 `2606.15032`（编号待复核，不挂链接） | — | — | — | — | — | — | — | L0–L7 证据阶梯；本文定位来源 |
| Lost in Aggregation `2606.22219` | — | — | — | — | — | — | — | 空间认知层级实测不单调；rollout_stage 不评分的佐证 |
| **EgoConseq-Bench（本文）** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 单张标定 ego 图 + 平面圆盘连续半径 + 度量程序 + 真实重 rollout |

外层结构：

```html
<!-- ================= 0.5 相关工作对比 ================= -->
<section id="related">
  <h2><small>Section 0.5 · Related Benchmarks</small>现有工作各占哪几格：conjunction 拆解</h2>
  <p class="lede">我们不主张任何单一维度是新的——每个维度都有先例。我们主张的是
  <b>受控合取（controlled conjunction）</b>：下表七列同时成立的组合，在我们对最近邻
  工作的定向核查中未找到相同者。逐行备注给出各工作的关键设计装置与我们借鉴的部分。</p>
  <div class="cmpwrap"><table class="cmp"> …按上表生成 14 行… </table></div>
  <div class="restraint"><b>冻结差异声明（契约 §3 原文）：</b>
  <i>"Widely used visual-spatial and intuitive-physics benchmarks usually treat
  the observer's embodiment as fixed or absent; recent body-aware navigation and
  collision benchmarks do not provide the same reset-matched continuous-radius
  consequence protocol."</i>
  <i>"In our directed comparison of the closest benchmarks, we did not find the
  same controlled combination: a metric body as a public, continuously intervened
  variable with true re-rollout; reset-matched do(action)/do(body)/do(sensor);
  three-layer ground truth separating physical fact, future observation, and
  visual support; and a preregistered operational visible-support publication
  contract."</i>
  措辞纪律：不做首创性宣称；贡献表述为 "we operationalize the conjunction"。</div>
</section>
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_pl_paper_page.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/egoconseq_paper.html tests/test_pl_paper_page.py
git commit -m "feat(paper-page): add related-benchmarks conjunction table"
```

---

### Task 3: 逐节"设计考虑"折叠框（S1/S2/S4/S5/S7/S8 各一个）

**Files:**
- Modify: `scripts/egoconseq_paper.html`
- Test: `tests/test_pl_paper_page.py`

**Interfaces:**
- Produces: `details.whybox` × 6；打印规则 `details.whybox{...}` 展开。

- [ ] **Step 1: Write the failing test**

```python
def test_paper_page_explains_design_rationale_per_section():
    html, parser, text = _page()
    whyboxes = [
        attrs for tag, attrs in parser.tags
        if tag == "details" and "whybox" in str(attrs.get("class", ""))
    ]
    assert len(whyboxes) >= 6
    for marker in (
        "为什么是单张图",
        "为什么是单一底物",
        "计分与选样机制",
        "为什么只讨论可视空间",
        "定位到感知层还是模拟层",
        "multi-modal gain",
    ):
        assert marker in text
    # 打印时折叠框内容必须可见
    assert "details.whybox" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_pl_paper_page.py::test_paper_page_explains_design_rationale_per_section -v`
Expected: FAIL

- [ ] **Step 3: Implement — CSS + 六个折叠框**

CSS 追加：

```css
details.whybox{margin:14px 0;border:1px solid var(--line,#e3ddd0);border-left:3px solid var(--accent,#8a5a2b);border-radius:8px;padding:10px 14px;font-size:14px;background:rgba(138,90,43,.04)}
details.whybox summary{cursor:pointer;font-weight:700;font-size:13px;letter-spacing:.04em}
details.whybox[open] summary{margin-bottom:8px}
```

`@media print` 块内追加（whybox 打印展开）：

```css
details.whybox{display:block}
details.whybox>*{display:block}
```

六个折叠框，统一结构 `<details class="whybox"><summary>设计考虑 · ……</summary><p>…</p></details>`，分别追加在对应 `</section>` 前：

1. **S1（#question）**：`<summary>设计考虑 · 为什么是单张图 + 度量程序</summary>`
   为什么是单张图：隔离"从一次观测做前向推演"的能力，排除视频记忆/多视角融合
   的混杂；也让可视支持契约可精确判定（witness 像素在这一张图上）。为什么是度量
   动作程序而非语言指令：消除指令歧义这个混杂变量——动作的语义由米数和角度唯一
   确定，答错只能归因于物理推演，不能推给理解偏差。
2. **S2（#substrate）**：`<summary>设计考虑 · 为什么是单一底物</summary>`
   四个头共享同一条 rollout，是联合一致性检验的前提（碰撞说"撞"而终点说"走完"
   即自相矛盾，可机械判定）；这区别于"拼盘式"题集——那里题目间不共享物理状态，
   无法交叉验证。四头 = forward model 输出的自然分解：交互 / 终点 / 未来观测 /
   可行性，正是任何 verify-before-act 栈需要的四个查询。
3. **S4（#interventions）**：`<summary>设计考虑 · 为什么干预不是题目分类，而是计分与选样机制</summary>`
   评测设计的通行纪律是两轴正交：能力轴（四个结果头）管可解释性——读者按
   "被预测对象"理解任务；对照轴（五种干预）管有效性——配对与联合计分把捷径
   策略的期望得分压回随机。行只描述预测对象、列只描述被改变量（契约 §5.3）。
   每种干预各自反驳一个具体质疑：do(action) 反驳"只看图猜"、do(body) 反驳
   "无视身体"、do(sensor) 反驳"分不清物理与视图"、信息消融反驳"标定没用"。
4. **S5（#contract）**：`<summary>设计考虑 · 为什么只讨论可视空间、只发布 sufficient</summary>`
   视野外空间的答案物理上确定但证据上不可判，把它出成题测的是先验不是推演；
   v1.1 裁决：证据充分性转为私有发布门（sufficient-only），不测弃权、公开选项
   中不含"证据不足"一类弃权项。三层 GT 分开存（物理事实 / 未来观测 / 可视支持），
   因为三者失败模式不同、审计路径不同，混存会让发布门无法独立判定。
5. **S7（#scoring）**：`<summary>设计考虑 · 为什么需要 oracle 梯子</summary>`
   总分只能说"模型不行"，梯子把失败定位到感知层还是模拟层：公平几何基线
   （Public Geometric Pipeline）回答"显式可算的部分模型算了没有"，
   Oracle-Depth / Oracle-Endpoint 逐级替换私有真值，掉分差即瓶颈位置。
6. **S8（#experiments）**：`<summary>设计考虑 · 审计实验的文献先例</summary>`
   数据集侧审计对应诊断型 benchmark 的标准论证：multi-modal gain / leakage
   报告名沿用 MMStar（arXiv 2403.20330）；盲基线 + 配对联合计分把"恒选同一
   答案"的期望得分压回随机，论证方式同 NaturalBench（arXiv 2410.14669）。
   我们只承诺机制，数字待正式评测产出后填入。

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_pl_paper_page.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/egoconseq_paper.html tests/test_pl_paper_page.py
git commit -m "feat(paper-page): add per-section design-rationale boxes"
```

---

### Task 4: Section 1 增强 — do(body) 翻转示意面板

**Files:**
- Modify: `scripts/egoconseq_paper.html`（#question 节内、`.formula` 之前插入）
- Test: `tests/test_pl_paper_page.py`

**Interfaces:**
- Produces: 容器 id `bodyflip`；CSS 类 `.discfig`。

- [ ] **Step 1: Write the failing test**

```python
def test_paper_page_shows_do_body_flip_panel():
    html, parser, text = _page()
    ids = {attrs["id"] for _tag, attrs in parser.tags if "id" in attrs}
    assert "bodyflip" in ids
    for marker in (
        "同一张图、同一动作，只换半径",
        "示意图，非真实案例",
        "oracle-v6 正式案例产出后替换",
    ):
        assert marker in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_pl_paper_page.py::test_paper_page_shows_do_body_flip_panel -v`
Expected: FAIL

- [ ] **Step 3: Implement — CSS 示意图 + 面板**

CSS 追加：

```css
#bodyflip{margin:18px 0;border:1px solid var(--line,#e3ddd0);border-radius:10px;padding:14px 16px}
.discfig{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px}
.discfig .lane{position:relative;height:96px;background:linear-gradient(90deg,#eee 0 12%,transparent 12% 88%,#eee 88%);border:1px solid var(--line,#e3ddd0);border-radius:8px}
.discfig .lane i{position:absolute;top:50%;transform:translateY(-50%);border-radius:50%;border:2px solid}
.discfig .lane.ok i{left:38%;width:34px;height:34px;border-color:#1c7c3c;background:rgba(28,124,60,.15)}
.discfig .lane.bad i{left:34%;width:64px;height:64px;border-color:#b3392e;background:rgba(179,57,46,.12)}
.discfig figcaption{font-size:12.5px;color:var(--muted,#6f6a5e);margin-top:6px}
@media(max-width:880px){.discfig{grid-template-columns:1fr}}
```

面板 HTML（插在 `.formula` div 之前）：

```html
<div id="bodyflip">
  <b>另一半故事：do(body) 翻转——同一张图、同一动作，只换半径，GT 翻转。</b>
  <p style="font-size:14px;margin:8px 0">上面的对照改的是动作；身体这一轴同样受控：
  身体是<b>公开的连续干预变量</b>，同一走廊、同一段前进，小半径圆盘保持间隙通过，
  大半径圆盘在同一位置切入障碍。GT 由连续临界半径二分搜索导出，公开网格上给出
  相邻半径转换区间（Q4，契约 §7.1）。</p>
  <div class="discfig">
    <figure><div class="lane ok"><i></i></div>
      <figcaption>r = 0.20 m：扫掠走廊全程保持间隙 → safe</figcaption></figure>
    <figure><div class="lane bad"><i></i></div>
      <figcaption>r = 0.30 m：同一动作在走廊收窄处接触 → collision</figcaption></figure>
  </div>
  <p style="font-size:12.5px;color:var(--muted,#6f6a5e);margin:8px 0 0">
  示意图，非真实案例（v131a preview 中 Q4 仅含 all_safe / all_collision 控制组，
  无翻转案例）；oracle-v6 正式案例产出后替换为真实图对。</p>
</div>
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_pl_paper_page.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/egoconseq_paper.html tests/test_pl_paper_page.py
git commit -m "feat(paper-page): add do(body) flip schematic to Section 1"
```

---

### Task 5: Section 3 扩展 — GT 流水线图 + 9 张 QA 设计卡

**Files:**
- Modify: `scripts/egoconseq_paper.html`（#examples 节内、现有 lede 段之后、截图组之前插入）
- Test: `tests/test_pl_paper_page.py`

**Interfaces:**
- Produces: 容器 id `gtpipe`、卡片类 `.qacard` × 9、网格 `.qacards`。

- [ ] **Step 1: Write the failing test**

```python
def test_paper_page_documents_qa_design_and_gt_derivation():
    html, parser, text = _page()
    ids = {attrs["id"] for _tag, attrs in parser.tags if "id" in attrs}
    assert "gtpipe" in ids
    qacards = [
        attrs for tag, attrs in parser.tags
        if "qacard" in str(attrs.get("class", ""))
    ]
    assert len(qacards) >= 9
    for marker in (
        "机械导出", "无人工标注", "无 LLM 标注",
        "圆盘沿动作程序扫掠",
        "签名法向带采样",
        "临界半径二分搜索",
        "半径无关",
        "yaw-only",
        "视锥测试 + 像素支持",
        "五个固定试探动作",
        "结果签名",
    ):
        assert marker in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_pl_paper_page.py::test_paper_page_documents_qa_design_and_gt_derivation -v`
Expected: FAIL

- [ ] **Step 3: Implement — CSS + 流水线 + 卡片**

CSS 追加：

```css
#gtpipe{margin:16px 0;padding:14px 16px;border:1px solid var(--line,#e3ddd0);border-radius:10px;background:rgba(28,124,60,.04)}
.qacards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0}
.qacard{border:1px solid var(--line,#e3ddd0);border-radius:10px;padding:12px 14px;font-size:13px}
.qacard b.qn{display:block;font-size:14px;margin-bottom:6px}
.qacard dl{margin:0}
.qacard dt{font-weight:700;font-size:11.5px;letter-spacing:.06em;color:var(--accent,#8a5a2b);margin-top:7px}
.qacard dd{margin:2px 0 0}
@media(max-width:880px){.qacards{grid-template-columns:1fr}}
```

流水线（插在 #examples 的 lede 之后）：

```html
<div id="gtpipe">
  <b>GT 是怎么来的：一条机械导出流水线，无人工标注、无 LLM 标注。</b>
  <p style="font-size:14px;margin:8px 0 0">
  输入五元组 (I, c, r, a, t) → 模拟器<b>真实重 rollout</b>：身体圆盘沿动作程序
  扫掠可行空间，碰撞即截断，记录实际执行前缀与终局位姿 → 三层 GT 分开存：
  <b>物理事实 GT</b>（碰撞/停止/终点/掩码）、<b>未来观测 GT</b>（终点重渲染的
  可见性）、<b>证据支持 GT</b>（witness 像素与可见性点集，私有发布门）→
  每族题目 = 该状态的一个投影读出。发布前统一过两道门：
  <span class="mono">execution_regime</span> 路由（Q6–Q9 主榜仅
  completed_clear）与 sufficient-only 可视支持契约。</p>
</div>
```

9 张卡（`<div class="qacards">` 内），每张四段 dt/dd（公开输入 / 题面与选项 /
GT 导出链 / 反捷径装置）。逐卡内容：

```html
<div class="qacards">
  <div class="qacard"><b class="qn">Q1 · 会不会碰</b><dl>
    <dt>公开输入</dt><dd>图 + 标定 + 单半径 + 动作</dd>
    <dt>题面与选项</dt><dd>执行该动作是否发生身体接触；safe / collision 二选一（forced choice）</dd>
    <dt>GT 导出链</dt><dd>圆盘沿动作程序扫掠可行空间，任一点接触即 collision</dd>
    <dt>反捷径</dt><dd>do(action)/do(body) 配对；标签 balance</dd></dl></div>
  <div class="qacard"><b class="qn">Q2 · 停在哪</b><dl>
    <dt>公开输入</dt><dd>同 Q1</dd>
    <dt>题面与选项</dt><dd>执行到第几段截断 + 实际执行米数（容差带内计分）</dd>
    <dt>GT 导出链</dt><dd>截断所在段索引 + 沿程实际位移，由同一次扫掠直接读出</dd>
    <dt>反捷径</dt><dd>度量读出：连续数值无法靠先验背答案</dd></dl></div>
  <div class="qacard"><b class="qn">Q3 · 首先撞到什么</b><dl>
    <dt>公开输入</dt><dd>同 Q1</dd>
    <dt>题面与选项</dt><dd>首次接触对象的超类；闭集选项（superclass primary）</dd>
    <dt>GT 导出链</dt><dd>接触世界点 → 世界→局部逆变换 → canonical 地板平面 → 签名法向带采样 → 语义归因（oracle v6）</dd>
    <dt>反捷径</dt><dd>超类闭集 + 归因消歧义门</dd></dl></div>
  <div class="qacard"><b class="qn">Q4 · 换身体会怎样</b><dl>
    <dt>公开输入</dt><dd>图 + 标定 + <b>四候选半径</b>（无单一当前半径）+ 动作</dd>
    <dt>题面与选项</dt><dd>四半径各自 safe/collision 的后果向量；相邻半径转换区间闭合</dd>
    <dt>GT 导出链</dt><dd>连续临界半径二分搜索 → 公开网格上 transition_rank；每半径真实重 rollout 验证</dd>
    <dt>反捷径</dt><dd>all_safe / all_collision 控制组：对"半径无关"的猜测策略召回随机分</dd></dl></div>
  <div class="qacard"><b class="qn">Q6 · 离目标更近了吗</b><dl>
    <dt>公开输入</dt><dd>图 + 标定 + 半径 + 动作 + 初始可见目标</dd>
    <dt>题面与选项</dt><dd>执行后目标距离趋势：closer / farther 闭合二值</dd>
    <dt>GT 导出链</dt><dd>目标参考集（私有完整几何点集）到圆盘中心的世界系米制距离 before/after 比较——半径无关定义</dd>
    <dt>反捷径</dt><dd>direction-heuristic-resistant 子集（点积符号启发式失败样本）单独报告</dd></dl></div>
  <div class="qacard"><b class="qn">Q7 · 目标在终点的方位</b><dl>
    <dt>公开输入</dt><dd>同 Q6</dd>
    <dt>题面与选项</dt><dd>终点 ego 系目标方位扇区：front / left / right / rear</dd>
    <dt>GT 导出链</dt><dd>实际终点位姿下的目标方位角 → 扇区离散化</dd>
    <dt>反捷径</dt><dd>必要性过滤器：sector(actual) ≠ sector(yaw-only)——只按转角猜（yaw-only 反事实）必错</dd></dl></div>
  <div class="qacard"><b class="qn">Q8 · 终点还看得见吗</b><dl>
    <dt>公开输入</dt><dd>同 Q6</dd>
    <dt>题面与选项</dt><dd>到达终点后目标是否仍在画面内；不可见时给消失原因（出视锥 / 被遮挡）</dd>
    <dt>GT 导出链</dt><dd>终点位姿重渲染：视锥测试 + 像素支持 ≥ 阈值；消失原因分层</dd>
    <dt>反捷径</dt><dd>FOV 对 = do(sensor) 型、遮挡对 = matched do(action) 型；witness 保留双方向</dd></dl></div>
  <div class="qacard"><b class="qn">Q9 · 终点还能动吗</b><dl>
    <dt>公开输入</dt><dd>图 + 标定 + 半径 + 动作</dd>
    <dt>题面与选项</dt><dd>终点处五个固定试探动作各自是否安全（5 位掩码）</dd>
    <dt>GT 导出链</dt><dd>终点位姿起步，五个固定试探动作各自扫掠 → 安全掩码</dd>
    <dt>反捷径</dt><dd>matched-mask 动作对 + 起/终点掩码必要性过滤器</dd></dl></div>
  <div class="qacard"><b class="qn">Q10 · 多约束选动作</b><dl>
    <dt>公开输入</dt><dd>图 + 标定 + 半径 + 四候选动作 + 约束集</dd>
    <dt>题面与选项</dt><dd>四候选中选满足全部约束者；独立 track，不进 headline</dd>
    <dt>GT 导出链</dt><dd>候选动作各自 rollout 的结果签名组合判定合法项</dd>
    <dt>反捷径</dt><dd>shortlist 强制含碰撞反例与约束 trade-off 项</dd></dl></div>
</div>
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_pl_paper_page.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/egoconseq_paper.html tests/test_pl_paper_page.py
git commit -m "feat(paper-page): add GT pipeline and per-family QA design cards"
```

---

### Task 6: 收尾 — 全量验证 + 浏览器/打印验收

**Files:**
- Modify: 无新增内容（仅修复本任务发现的问题）
- Test: 全量

- [ ] **Step 1: 全量 pytest**

Run: `$PY -m pytest -q`
Expected: 0 failures（基线 1316+ 项，新增 5 项）

- [ ] **Step 2: 措辞红线 grep 复核（人工看上下文）**

```bash
grep -n "首个\|没有先例\|the first" scripts/egoconseq_paper.html
```
Expected: 无输出，或仅出现在否定语境（如"不使用'首个'"）——测试已断言，此处人工确认。

- [ ] **Step 3: 浏览器验收（8770 已运行则直接刷新）**

打开 `http://127.0.0.1:8770/scripts/egoconseq_paper.html`：
- Section 0 / 0.5 渲染正常，对比表 14 行、窄窗口容器内横向滚动；
- 6 个折叠框展开收起正常；#bodyflip 两泳道示意显示；9 张卡三列网格、
  ≤880px 单列；
- 打印预览（Ctrl+P）：whybox 内容可见、CTA 隐藏。

- [ ] **Step 4: 最终提交（如 Step 2/3 产生修复）**

```bash
git add scripts/egoconseq_paper.html
git commit -m "fix(paper-page): closeout tweaks from browser/print acceptance"
```

---

## Self-Review 记录

- Spec 覆盖：①→Task 1，②→Task 2，③→Task 3，④→Task 4，⑤(a)(b)→Task 5，⑥→各任务内嵌 + Task 6。无缺口。
- 占位符扫描：Task 2 表格以逐格矩阵给出（"…按上表生成 14 行…"指代同任务内的完整矩阵，非外部引用）。
- 一致性：类名 `.whybox/.qacard/.cmp/.discfig`、id `why/related/bodyflip/gtpipe` 全程一致；测试标记字符串与实现 HTML 逐字对应。
