/* Published QA is read-only. Review records retain the original release-scoped storage keys. */
"use strict";
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = n => n.toLocaleString("en-US");
const TASKS = DATA.statistics.taxonomy.tasks;
const DOMAINS = DATA.statistics.taxonomy.domains;
const REPS = {atomic:"原子动作",merged:"合并动作",coarse:"粗粒度动作"};
const SETS = {benchmark:"主 benchmark",radius_pairs:"半径对照集"};
const PAGE = 12;
const reviewKey = r => r.review_key || r.release_sha256;
const KEYS = DATA.releases.map(r => "review:" + reviewKey(r));
const state = KEYS.map(k => {try {const s = JSON.parse(localStorage.getItem(k) || "{}"); return s && typeof s === "object" && !Array.isArray(s) ? s : {};} catch {return {};}});
const FACETS = [["release","评测分区"],["set","题集"],["dataset","数据集"],["domain","后果域"],["task","任务"],["scene","场景"],["rep","动作表示"],["unit","长度单位"],["answer","答案 / 距离档"],["status","审阅状态"]];
const QUESTIONS = [["answerable","能独立作答"],["visible","决定答案的信息可见"],["multiple","有多个合理答案"],["conflict","题干 / 图片 / 答案冲突"]];
const REASONS = ["无分歧","推理失误","信息不足","题意歧义","GT/资产问题"];
const selects = {};
let page = 0, filtered = [], view = "browse", showAnswers = true, toastTimer, searchTimer;
let chartSelection = null, chartLabel = '';
const flagged = s => s.answerable === "否" || s.visible === "否" || s.multiple === "是" || s.conflict === "是" || ["信息不足","题意歧义","GT/资产问题"].includes(s.reason);
const status = it => {const s = state[it.r][it.id]; return !s || !Object.values(s).some(Boolean) ? "未记录" : flagged(s) ? "有问题" : "有记录";};
const value = (it,f) => f === "release" ? DATA.releases[it.r].name : f === "status" ? status(it) : String(it[f] ?? "");
const label = (f,v) => f === "task" ? TASKS[v] : f === "domain" ? DOMAINS[v].short : f === "rep" ? REPS[v] || v : f === "set" ? SETS[v] || v : f === "dataset" ? v.toUpperCase() : v;
const itemKey = it => `${it.r}:${it.id}`;
const byKey = new Map(DATA.items.map(it => [itemKey(it),it]));
const searchText = new Map(DATA.items.map(it => [itemKey(it),`${it.id} ${it.obs} ${it.scene} ${it.pair} ${it.program} ${TASKS[it.task]} ${DOMAINS[it.domain].name} ${it.prompt}`.toLowerCase()]));
function hash(s) {let n = 2166136261; for (const c of s) n = Math.imul(n ^ c.charCodeAt(0),16777619); return n >>> 0;}
// Only the display order changes: rotate tasks so a row contains different kinds of cases.
const buckets = new Map(Object.keys(TASKS).map(t => [t,[]]));
for (const it of DATA.items) buckets.get(it.task).push(it);
for (const rows of buckets.values()) rows.sort((a,b) => hash(itemKey(a))-hash(itemKey(b)) || itemKey(a).localeCompare(itemKey(b)));
const displayOrder = [];
for (let n = 0; displayOrder.length < DATA.items.length; n++) for (const t of ["A1","A4","C1","A2","B1","B2","A3","B3"]) if (buckets.get(t)[n]) displayOrder.push(buckets.get(t)[n]);

function notify(message) {$('toast').textContent = message; $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true,4500);}
function save(r) {try {localStorage.setItem(KEYS[r],JSON.stringify(state[r]));} catch {notify("浏览器未能保存记录，请立即导出审阅 JSON 备份。");}}
function matches(it,skip) {
  if (chartSelection && !chartSelection.has(itemKey(it))) return false;
  if ($('sample').checked && !it.sample) return false;
  const q = $('q').value.trim().toLowerCase();
  if (q && !searchText.get(itemKey(it)).includes(q)) return false;
  return FACETS.every(([f]) => f === skip || !selects[f].value || value(it,f) === selects[f].value);
}
for (const [i,[f,name]] of FACETS.entries()) {
  const el = document.createElement("label"); el.className = "facet";
  el.innerHTML = `<span>${name}</span><select aria-label="${name}"><option value="">全部</option></select>`;
  selects[f] = el.querySelector("select");
  selects[f].addEventListener("change",() => {page = 0; render();});
  $(i < 5 ? 'primary-facets' : 'secondary-facets').append(el);
}
function facets() {
  // Preserve selections even when their combination has no results; never silently clear a filter.
  const current = Object.fromEntries(FACETS.map(([f]) => [f,selects[f].value]));
  for (const [f] of FACETS) {
    const c = new Map();
    for (const it of DATA.items) if (matches(it,f)) c.set(value(it,f),(c.get(value(it,f))||0)+1);
    if (current[f] && !c.has(current[f])) c.set(current[f],0);
    selects[f].innerHTML = '<option value="">全部</option>' + [...c].sort((a,b) => a[0].localeCompare(b[0],"zh",{numeric:true})).map(([v,n]) => `<option value="${esc(v)}">${esc(label(f,v))} · ${fmt(n)}</option>`).join("");
    selects[f].value = current[f];
  }
}
function imageMarkup(it,index,caption) {
  if (!it.images[index]) return '<span class="image-error">该图片引用缺失</span>';
  return `<button class="image-button" data-image="${index}" aria-label="放大${esc(caption)}"><img src="${esc(it.images[index])}" alt="${esc(caption)}" loading="lazy" decoding="async" width="640" height="480"></button>`;
}
function promptMarkup(it) {
  const lines = it.prompt.split("\n"), out = []; let imageIndex = 0;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i], trimmed = line.trim();
    if (!trimmed) continue;
    if (trimmed === "Configuration:") {
      const cfg = [];
      while (i+1 < lines.length && /^\s*- /.test(lines[i+1])) {
        const raw = lines[++i].trim(), split = raw.indexOf(":");
        cfg.push(split < 0 ? `<div class="config-line">${esc(raw)}</div>` : `<div class="config-line"><span>${esc(raw.slice(0,split+1))}</span><strong>${esc(raw.slice(split+1).trim())}</strong></div>`);
      }
      out.push(`<div class="section-label">${esc(line)}</div><div class="configuration">${cfg.join("")}</div>`); continue;
    }
    if (trimmed === "Actions:") {
      const actions = [];
      while (i+1 < lines.length && /^\s*\(\d+\)/.test(lines[i+1])) {
        const m = lines[++i].match(/^\s*(\(\d+\))\s*(.*)$/);
        actions.push(`<li><span class="step-number">${esc(m[1])}</span><span>${esc(m[2])}</span></li>`);
      }
      out.push(`<div class="section-label">${esc(line)}</div><ol class="actions-list">${actions.join("")}</ol>`); continue;
    }
    if (/^[A-D]\.\s*<image>\s*$/.test(trimmed)) {
      const options = [];
      do {
        const letter = lines[i].trim()[0];
        options.push(`<figure class="option-figure">${imageMarkup(it,imageIndex++,"选项 "+letter)}<figcaption class="option-label">${letter}.</figcaption></figure>`);
        if (i+1 >= lines.length || !/^[A-D]\.\s*<image>\s*$/.test(lines[i+1].trim())) break;
        i++;
      } while (true);
      out.push(`<div class="options-grid">${options.join("")}</div>`); continue;
    }
    if (line.includes("<image>")) {
      const parts = line.split("<image>");
      parts.forEach((p,n) => {if (p) out.push(`<p class="prompt-line">${esc(p)}</p>`); if (n < parts.length-1) out.push(imageMarkup(it,imageIndex++,"初始图"));}); continue;
    }
    const kind = /^Question:/.test(trimmed) ? "question" : /^Answer\b/.test(trimmed) ? "instruction" : /^(A collision occurs|The action sequence is collision-free)/.test(trimmed) ? "premise" : /initial view:$/i.test(trimmed) ? "initial-title" : "";
    out.push(`<p class="prompt-line ${kind}">${esc(line)}</p>`);
  }
  return out.join("");
}
function reviewMarkup(it) {
  const s = state[it.r][it.id] || {};
  return `<div class="review-form"><label class="review-field">我的答案<input data-field="mine" type="text" value="${esc(s.mine||'')}" placeholder="可先隐藏标准答案，再独立作答"></label><label class="review-field">判断依据<input data-field="basis" type="text" value="${esc(s.basis||'')}" placeholder="画面中的哪些信息支持判断？"></label>` + QUESTIONS.map(([f,n]) => `<div class="review-choice"><span>${n}</span><div>${["是","否"].map(v => `<button data-field="${f}" data-choice="${v}" aria-pressed="${s[f]===v}">${v}</button>`).join("")}</div></div>`).join("") + `<label class="review-field">分歧原因<select data-field="reason"><option value="">未选择</option>${REASONS.map(v => `<option ${s.reason===v?'selected':''}>${v}</option>`).join("")}</select></label><label class="review-field">备注<textarea data-field="note" placeholder="记录问题，或值得保留的例子">${esc(s.note||'')}</textarea></label><div class="review-foot">修改后自动保存到本浏览器 · 有记录不等于已完整审阅</div></div>`;
}
function card(it) {
  const s = status(it), taskClass = ["A4","B1","B2","B3"].includes(it.task) ? "spatial" : it.task === "C1" ? "visual" : "";
  return `<article class="case-card" data-key="${esc(itemKey(it))}" aria-label="${esc(TASKS[it.task]+' '+it.id)}"><header class="card-head"><div class="task-label"><span class="task-code ${taskClass}">${DOMAINS[it.domain].short}</span><span class="task-name">${TASKS[it.task]}</span></div><span class="dataset-tag">${esc(it.dataset.toUpperCase())}</span></header><div class="card-context"><span class="scene-name">${esc(it.scene)}</span><span class="mini-dot"></span><span>${esc(DATA.releases[it.r].name)}</span><span class="pill">${esc(REPS[it.rep]||it.rep)}</span><span class="pill unit">${esc(it.unit)}</span>${it.set==='radius_pairs'?'<span class="pill">半径对照</span>':''}${it.sample?'<span class="pill">抽查样本</span>':''}</div><div class="prompt-content">${promptMarkup(it)}</div><div class="answer-box"><span class="answer-caption">标准答案 · GT</span><strong class="answer-value">${esc(it.gt)}</strong><span class="answer-hidden">已隐藏 · 顶部按钮可显示</span></div>${it.pair?'<button class="pair-button" data-pair="'+esc(it.pair)+'">查看这组半径对照 ↗</button>':''}<div class="card-tools"><details><summary>记录审阅<span class="review-state ${s==='有问题'?'flag':''}">${s}</span></summary>${reviewMarkup(it)}</details><details class="evidence"><summary>查看采集证据</summary><ul class="facts">${it.facts.map(f=>'<li>'+esc(f)+'</li>').join('')}</ul><p class="evidence-note">辅助证据不等于可回答性结论。</p></details><details><summary>系统提示与原始题干</summary><div class="section-label">SYSTEM</div><pre class="system-text">${esc(DATA.systems[it.sys])}</pre><div class="section-label">USER · 原文（&lt;image&gt; 对应上方图像）</div><pre class="source-prompt">${esc(it.prompt)}</pre></details></div><div class="item-id">${esc(it.id)}</div></article>`;
}
function updateDone() {const n = DATA.items.filter(it => status(it)!=="未记录").length; $('done').textContent = `${fmt(n)} 题有记录`;}
function render() {
  facets(); filtered = displayOrder.filter(it => matches(it));
  const pages = Math.max(1,Math.ceil(filtered.length/PAGE)); page = Math.min(page,pages-1);
  const scenes = new Set(filtered.map(it=>it.dataset+":"+it.scene)).size;
  const observations = new Set(filtered.map(it=>it.dataset+":"+it.obs)).size;
  $('totals').innerHTML = [[filtered.length,"当前题目"],[observations,"独立观测"],[scenes,"场景"],[new Set(filtered.map(it=>it.task)).size,"任务类型"]].map(([n,t])=>`<div class="metric"><strong>${fmt(n)}</strong><span>${t}</span></div>`).join("");
  const active = FACETS.filter(([f])=>selects[f].value).map(([f])=>label(f,selects[f].value));
  $('scope').innerHTML = `当前范围：<b>${esc(active.join(" / ")||"全部已发布题目")}</b>${$('sample').checked?' · 分层抽查样本':''}${$('q').value?' · 搜索结果':''}${chartSelection?' · 图表选中：'+esc(chartLabel):''}`;
  $('count').textContent = filtered.length ? `${fmt(filtered.length)} 题 · 本页 ${page*PAGE+1}–${Math.min((page+1)*PAGE,filtered.length)}` : "0 题";
  for (const id of ['page-top','page-bottom']) $(id).textContent = `${page+1} / ${pages}`;
  for (const id of ['prev','prev-bottom']) $(id).disabled = page===0;
  for (const id of ['next','next-bottom']) $(id).disabled = page>=pages-1;
  $('items').innerHTML = filtered.slice(page*PAGE,(page+1)*PAGE).map(card).join("");
  $('empty').hidden = filtered.length>0;
  updateDone(); if (view==='stats') renderStats();
}
function renderStats() {Stats.render(filtered);}
function switchView(next) {view=next; $('browse-view').hidden=next!=='browse'; $('stats-view').hidden=next!=='stats'; for (const [id,v] of [['browse-tab','browse'],['stats-tab','stats']]) {$(id).classList.toggle('active',next===v); $(id).setAttribute('aria-pressed',String(next===v));} history.replaceState(null,'',next==='stats'?'#stats':'#browse'); if(next==='stats') renderStats();}
function reset() {for(const s of Object.values(selects)) s.value=''; chartSelection=null;chartLabel=''; $('q').value=''; $('sample').checked=false; page=0; render();}
function changePage(delta) {page += delta; render(); $('browse-view').scrollIntoView({block:'start'});}
for(const id of ['prev','prev-bottom']) $(id).onclick=()=>changePage(-1);
for(const id of ['next','next-bottom']) $(id).onclick=()=>changePage(1);
$('q').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{page=0;render();},180);};
$('sample').onchange=()=>{page=0;render();}; $('reset').onclick=reset; $('empty-reset').onclick=reset;
$('browse-tab').onclick=()=>switchView('browse'); $('stats-tab').onclick=()=>switchView('stats');
$('more-filters').onclick=()=>{const hidden=!$('advanced').hidden;$('advanced').hidden=hidden;$('more-filters').setAttribute('aria-expanded',String(!hidden));$('more-filters').innerHTML=`更多筛选 <span>${hidden?'＋':'−'}</span>`;};
$('answers-toggle').onclick=()=>{showAnswers=!showAnswers;document.body.classList.toggle('hide-answers',!showAnswers);$('answers-toggle').textContent=showAnswers?'答案已显示':'答案已隐藏';$('answers-toggle').setAttribute('aria-pressed',String(showAnswers));};
document.querySelectorAll('[data-columns]').forEach(b=>b.onclick=()=>{document.documentElement.style.setProperty('--columns',b.dataset.columns);document.querySelectorAll('[data-columns]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});});
function writeReview(el) {
  const root=el.closest('.case-card'), it=byKey.get(root.dataset.key), s=state[it.r][it.id] ||= {};
  s[el.dataset.field]=el.dataset.choice || el.value;
  if(el.dataset.choice) root.querySelectorAll(`button[data-field="${el.dataset.field}"]`).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.choice===s[el.dataset.field])));
  save(it.r); const tag=root.querySelector('.review-state');tag.textContent=status(it);tag.classList.toggle('flag',flagged(s));updateDone();
}
$('items').addEventListener('input',e=>{if(e.target.matches('input[data-field],textarea[data-field]'))writeReview(e.target);});
$('items').addEventListener('change',e=>{if(e.target.matches('select[data-field]'))writeReview(e.target);});
$('items').addEventListener('error',e=>{if(e.target.tagName==='IMG'){e.target.alt='图片加载失败：'+e.target.alt;e.target.closest('.image-button').setAttribute('title','图片加载失败，请检查服务目录');}},true);
let lightboxItem=null, lightboxIndex=0;
function showImage(it,index) {
  lightboxItem=it; lightboxIndex=index;
  const caption=index===0?'初始图':'选项 '+String.fromCharCode(64+index);
  $('lightbox-title').textContent=`${it.dataset.toUpperCase()} · ${TASKS[it.task]} · ${it.scene} / ${caption}`;
  $('lightbox-image').src=it.images[index]; $('lightbox-image').alt=caption;
  $('image-original').href=it.images[index]; $('image-counter').textContent=`${index+1} / ${it.images.length}`;
  $('image-prev').disabled=index===0; $('image-next').disabled=index===it.images.length-1;
  if(!$('lightbox').open)$('lightbox').showModal();
}
$('items').addEventListener('click',e=>{
  const image=e.target.closest('[data-image]'), choice=e.target.closest('button[data-field]'), pair=e.target.closest('[data-pair]');
  if(image)showImage(byKey.get(image.closest('.case-card').dataset.key),Number(image.dataset.image));
  if(choice)writeReview(choice);
  if(pair){reset();selects.set.value='radius_pairs';$('q').value=pair.dataset.pair;page=0;render();$('browse-view').scrollIntoView({block:'start'});}
});
$('lightbox-close').onclick=()=>$('lightbox').close();
$('lightbox').addEventListener('click',e=>{if(e.target===$('lightbox'))$('lightbox').close();});
$('image-prev').onclick=()=>showImage(lightboxItem,lightboxIndex-1);
$('image-next').onclick=()=>showImage(lightboxItem,lightboxIndex+1);
document.addEventListener('keydown',e=>{if(!$('lightbox').open || !['ArrowLeft','ArrowRight'].includes(e.key))return;const i=lightboxIndex+(e.key==='ArrowLeft'?-1:1);if(i>=0&&i<lightboxItem.images.length){e.preventDefault();showImage(lightboxItem,i);}});
function drillChart(target) {
  const cell=target.closest('[data-drill]');if(!cell)return;
  const group=Stats.drill(Number(cell.dataset.drill));if(!group?.rows.length)return;
  chartSelection=new Set(group.rows.map(itemKey));chartLabel=group.label;page=0;
  // Some comparison charts intentionally include both releases. Keep exactly the selected rows.
  for(const s of Object.values(selects))s.value='';$('q').value='';$('sample').checked=false;
  switchView('browse');render();$('browse-view').scrollIntoView({block:'start'});
}
$('stats-content').addEventListener('click',e=>drillChart(e.target));
$('stats-content').addEventListener('keydown',e=>{if(['Enter',' '].includes(e.key)&&e.target.matches('rect[data-drill],path[data-drill]')){e.preventDefault();drillChart(e.target);}});
$('export').onclick=()=>{
  const reviews=Object.fromEntries(DATA.releases.map((r,i)=>[reviewKey(r),state[i]]));
  const blob=new Blob([JSON.stringify({exported_utc:new Date().toISOString(),releases:DATA.releases,reviews},null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='pbench-review-'+new Date().toISOString().slice(0,10)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
$('import').onchange=async e=>{
  const file=e.target.files[0];if(!file)return;
  try{
    const data=JSON.parse(await file.text());if(!data.reviews||typeof data.reviews!=='object')throw new Error('缺少 reviews 字段');let n=0;
    DATA.releases.forEach((r,i)=>{const incoming=data.reviews[reviewKey(r)];if(!incoming)return;for(const[id,record]of Object.entries(incoming)){if(!byKey.has(`${i}:${id}`)||!record||typeof record!=='object')continue;const safe={};for(const f of ['mine','basis','answerable','visible','multiple','conflict','reason','note'])if(typeof record[f]==='string')safe[f]=record[f];state[i][id]={...state[i][id],...safe};n++;}save(i);});
    render();notify(n?`已导入 ${n} 条审阅记录。`:'没有匹配当前发布版本的记录，原记录未改动。');
  }catch(err){notify('导入失败：'+err.message);}finally{e.target.value='';}
};
$('version').textContent='当前 Benchmark';
facets();selects.release.value=DATA.releases[0]?.name||'';selects.set.value='benchmark';render();
if(location.hash==='#stats'||location.hash.startsWith('#stat-'))switchView('stats');
