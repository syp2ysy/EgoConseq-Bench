/* Research-backed, local-only charts. Every aggregate is computed from the current release items. */
"use strict";
const Stats = (() => {
  const DOMAINS = DATA.statistics?.taxonomy.domains || {};
  const SOURCES = DATA.statistics?.sources || {};
  const HORIZONTAL = ["front","front-right","right","rear-right","rear","rear-left","left","front-left"];
  const HN = ["正前","右前","正右","右后","正后","左后","正左","左前"];
  const VERTICAL = ["above","level","below"], VN = ["上方","水平","下方"];
  const COLORS = ["#23624d","#6266a5","#ad633d","#649da1","#bb9b60","#9d7c9b"];
  let drills = [], panels = 0;
  const finite = x => typeof x === "number" && Number.isFinite(x);
  const pct = (n,d) => d ? (100*n/d).toFixed(1)+"%" : "—";
  const mean = a => a.length ? a.reduce((s,x)=>s+x,0)/a.length : null;
  const unique = (rows,key) => [...new Map(rows.map(i=>[key(i),i])).values()];
  const numeric = (rows,key) => rows.map(i=>i.m?.[key]).filter(finite);
  const taskRows = (rows,...tasks) => rows.filter(i=>tasks.includes(i.task));
  const obsRows = rows => unique(rows,i=>`${i.dataset}/${i.obs}`);
  const dsRows = rows => [...new Set(rows.map(i=>i.dataset))].sort();
  const ordinal = a => [...a].sort((x,y)=>Number(x)-Number(y));
  function tally(rows,get) {const m=new Map();for(const i of rows){const k=get(i);m.set(k,(m.get(k)||0)+1);}return [...m].sort((a,b)=>b[1]-a[1]||String(a[0]).localeCompare(String(b[0])));}
  function quantile(values,p) {if(!values.length)return null;const a=[...values].sort((x,y)=>x-y),x=(a.length-1)*p,l=Math.floor(x);return a[l]+(a[Math.ceil(x)]-a[l])*(x-l);}
  function bind(rows,label) {drills.push({rows:rows.flatMap(i=>i._items||[i]),label});return drills.length-1;}
  function cite(ids) {return '<p class="chart-source">参考方法 · '+ids.map(id=>`<a href="${esc(SOURCES[id]?.url)}" target="_blank" rel="noopener">${esc(SOURCES[id]?.name)}</a>`).join(' / ')+'</p>';}
  function panel(title,caption,body,refs=[],wide=false) {panels++;return `<section class="stats-panel ${wide?'wide':''}"><h3>${esc(title)}</h3><p class="stats-caption">${esc(caption)}</p>${body}${refs.length?cite(refs):''}</section>`;}
  function section(id,title,caption,body) {return `<section class="statistics-section" id="stat-${id}"><div class="section-heading"><span>${esc(id.toUpperCase())}</span><div><h2>${esc(title)}</h2><p>${esc(caption)}</p></div><a href="#stats-top" aria-label="返回统计目录">↑</a></div><div class="stats-grid">${body}</div></section>`;}
  const empty = message => `<p class="chart-empty">${esc(message || '当前筛选没有适用题目')}</p>`;
  function table(head,rows) {return `<div class="stat-table-wrap"><table class="stat-table"><thead><tr>${head.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map((v,i)=>`<${i?'td':'th'}>${esc(v??'—')}</${i?'td':'th'}>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
  function metrics(rows) {return `<div class="stat-metrics">${rows.map(([v,k])=>`<div><strong>${esc(v)}</strong><span>${esc(k)}</span></div>`).join('')}</div>`;}
  function summary(rows) {
    const programs=tally(rows,i=>i.program),scenes=tally(rows,i=>i.dataset+'/'+i.scene);
    return {questions:rows.length,observations:obsRows(rows).length,scenes:scenes.length,
      initial_images:new Set(rows.map(i=>i.raw_hash)).size,input_images:new Set(rows.flatMap(i=>i.image_hashes)).size,
      pairs:new Set(rows.filter(i=>i.pair).map(i=>i.pair)).size,programs:programs.length,
      image_references:rows.reduce((s,i)=>s+i.images.length,0),program_repeat_excess:rows.length-programs.length,
      top10_program_questions:programs.slice(0,10).reduce((s,x)=>s+x[1],0),max_scene_questions:scenes[0]?.[1]||0};
  }
  function bars(rows,get,{order=null,names=x=>x,limit=0,color=COLORS[0]}={}) {
    let groups=tally(rows,get);
    if(order)groups=order.map(k=>[k,groups.find(x=>x[0]===k)?.[1]||0]);
    const all=groups; if(limit)groups=groups.slice(0,limit);
    const max=Math.max(1,...groups.map(x=>x[1]));
    const output=groups.map(([key,n])=>`<button class="stat-bar" data-drill="${bind(rows.filter(i=>get(i)===key),String(names(key)))}" ${n?'':'disabled'} title="${esc(names(key))}：${n} / ${rows.length}"><span class="stat-bar-name">${esc(names(key))}</span><span class="stat-bar-track"><span style="width:${100*n/max}%;background:${color}"></span></span><span class="stat-bar-value">${fmt(n)}<small>${pct(n,rows.length)}</small></span></button>`).join('');
    return (rows.length?`<div class="stat-bars">${output}</div>`:empty())+(limit&&all.length>limit?`<details class="chart-details"><summary>全部 ${fmt(all.length)} 项 · 当前展示前 ${limit} 项</summary>${table(['取值','题数','占比'],all.map(([k,n])=>[names(k),fmt(n),pct(n,rows.length)]))}</details>`:'');
  }
  function histogram(rows,key,edges,unit='',color=COLORS[0]) {
    const valid=rows.filter(i=>finite(i.m?.[key]));
    const groups=edges.slice(0,-1).map((lo,j)=>({lo,hi:edges[j+1],rows:valid.filter(i=>i.m[key]>=lo&&(i.m[key]<edges[j+1]||(j===edges.length-2&&i.m[key]===edges[j+1])))}));
    const labels=groups.map(g=>!Number.isFinite(g.lo)?`<${g.hi}`:!Number.isFinite(g.hi)?`≥${g.lo}`:`${g.lo}–${g.hi}`);
    if(!valid.length)return empty();
    const max=Math.max(1,...groups.map(g=>g.rows.length)),width=650,height=235,left=42,bottom=38,top=20,plotH=height-top-bottom,bw=(width-left-12)/groups.length;
    const ticks=[0,.25,.5,.75,1].map(f=>`<line x1="${left}" x2="${width-10}" y1="${top+plotH*(1-f)}" y2="${top+plotH*(1-f)}" stroke="#e7ebe4"/><text x="${left-7}" y="${top+plotH*(1-f)+4}" text-anchor="end">${Math.round(max*f)}</text>`).join('');
    const rects=groups.map((g,j)=>{const n=g.rows.length,x=left+j*bw+5,h=n/max*plotH;return `<g><rect x="${x}" y="${top+plotH-h}" width="${Math.max(2,bw-10)}" height="${h}" rx="3" fill="${color}" opacity=".86" ${n?`tabindex="0" role="button" data-drill="${bind(g.rows,labels[j]+' '+unit)}" aria-label="${esc(labels[j]+' '+unit+'，'+n+' 个样本')}"`:''}><title>${esc(labels[j])} ${unit}：${n}（${pct(n,valid.length)}）</title></rect><text x="${x+(bw-10)/2}" y="${height-bottom+19}" text-anchor="middle">${esc(labels[j])}</text></g>`;}).join('');
    const values=valid.map(i=>i.m[key]),dec=unit==='°'?1:2;
    return `<svg class="histogram" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(key)} 分档计数，${valid.length} 个有效样本">${ticks}${rects}<text x="${width-12}" y="${height-4}" text-anchor="end">${esc(unit)}</text></svg><p class="distribution-note">N = ${fmt(valid.length)} · 中位数 ${quantile(values,.5).toFixed(dec)} ${esc(unit)} · P10–P90 ${quantile(values,.1).toFixed(dec)}–${quantile(values,.9).toFixed(dec)} ${esc(unit)}${valid.length<rows.length?' · 缺失 '+fmt(rows.length-valid.length):''}</p>`;
  }
  function matrix(rows,rowKeys,colKeys,rowGet,colGet,rowName=x=>x,colName=x=>x,unavailable=()=>false) {
    const groups=rowKeys.map(r=>colKeys.map(c=>rows.filter(i=>rowGet(i)===r&&colGet(i)===c)));
    const max=Math.max(1,...groups.flat().map(g=>g.length));
    return `<div class="matrix-wrap"><table class="matrix"><thead><tr><th></th>${colKeys.map(c=>`<th>${esc(colName(c))}</th>`).join('')}</tr></thead><tbody>${rowKeys.map((r,ri)=>`<tr><th>${esc(rowName(r))}</th>${colKeys.map((c,ci)=>unavailable(r,c)?'<td class="not-applicable" title="对应 meta 任务表为空或不存在">无供给</td>':`<td><button data-drill="${bind(groups[ri][ci],rowName(r)+' / '+colName(c))}" style="--heat:${Math.round(groups[ri][ci].length/max*43)}%" ${groups[ri][ci].length?'':'disabled'}>${fmt(groups[ri][ci].length)}</button></td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  function stacked(rows,groupGet,categoryGet,categories,names=x=>x) {
    return '<div class="stacked-chart">'+tally(rows,groupGet).map(([g,n])=>{const subset=rows.filter(i=>groupGet(i)===g);return `<div class="stacked-row"><span>${esc(names(g))}<small>N=${fmt(n)}</small></span><div class="stacked-track">${categories.map(([key,name,color])=>{const part=subset.filter(i=>categoryGet(i)===key);return part.length?`<button style="width:${part.length/n*100}%;background:${color}" data-drill="${bind(part,names(g)+' / '+name)}" title="${esc(name)}：${part.length} / ${n}（${pct(part.length,n)}）">${part.length/n>.12?pct(part.length,n):''}</button>`:'';}).join('')}</div></div>`;}).join('')+`<div class="chart-legend">${categories.map(([,n,c])=>`<span><i style="background:${c}"></i>${esc(n)}</span>`).join('')}</div></div>`;
  }
  function taxonomy(rows) {
    if(!rows.length)return empty();
    const cx=180,cy=180,tau=2*Math.PI;let angle=-Math.PI/2;
    function arc(start,end,r0,r1) {end=Math.min(end,start+tau-1e-6);const xy=(r,a)=>[cx+r*Math.cos(a),cy+r*Math.sin(a)],p=[xy(r1,start),xy(r1,end),xy(r0,end),xy(r0,start)],big=end-start>Math.PI?1:0;return `M${p[0]} A${r1},${r1} 0 ${big} 1 ${p[1]} L${p[2]} A${r0},${r0} 0 ${big} 0 ${p[3]} Z`;}
    let shapes='',legend='';
    for(const [key,d] of Object.entries(DOMAINS)){
      const subset=rows.filter(i=>i.domain===key),end=angle+tau*subset.length/rows.length;if(!subset.length)continue;
      shapes+=`<path d="${arc(angle,end,81,118)}" fill="${d.color}" stroke="white" stroke-width="2" tabindex="0" role="button" data-drill="${bind(subset,d.name)}" aria-label="${esc(d.name)} ${subset.length} 题"><title>${esc(d.name)}：${subset.length}（${pct(subset.length,rows.length)}）</title></path>`;
      let a=angle;legend+=`<div class="taxonomy-domain"><h4><i style="background:${d.color}"></i>${esc(d.name)}<span>${fmt(subset.length)}</span></h4>`;
      for(const [j,t] of d.tasks.entries()) {const part=subset.filter(i=>i.task===t),b=a+tau*part.length/rows.length;if(part.length)shapes+=`<path d="${arc(a,b,123,162)}" fill="${d.color}" opacity="${.55+.4*(j+1)/d.tasks.length}" stroke="white" stroke-width="2" tabindex="0" role="button" data-drill="${bind(part,TASKS[t])}" aria-label="${esc(TASKS[t])} ${part.length} 题"><title>${esc(TASKS[t])}：${part.length}</title></path>`;legend+=`<button data-drill="${bind(part,TASKS[t])}" ${part.length?'':'disabled'}><span>${TASKS[t]}</span><b>${fmt(part.length)}</b><small>${pct(part.length,rows.length)}</small></button>`;a=b;}
      legend+='</div>';angle=end;
    }
    return `<div class="taxonomy-layout"><svg viewBox="0 0 360 360" role="img" aria-label="三个后果域与八项任务的题量分布">${shapes}<text x="180" y="171" class="donut-number" text-anchor="middle">${fmt(rows.length)}</text><text x="180" y="198" text-anchor="middle" class="donut-caption">当前范围 · 题</text></svg><div class="taxonomy-legend">${legend}</div></div>`;
  }
  function conditional(rows,get,name) {
    return table([name,'碰撞 / 总题数','碰撞率'],tally(rows,get).map(([key,n])=>{const part=rows.filter(i=>get(i)===key),hits=part.filter(i=>i.m.collision).length;return [key,`${hits} / ${n}`,pct(hits,n)];}));
  }
  function datasetFacet(rows,render) {return '<div class="dataset-facets">'+dsRows(rows).map(d=>`<div class="dataset-facet"><h4>${d.toUpperCase()}</h4>${render(rows.filter(i=>i.dataset===d),d)}</div>`).join('')+'</div>';}
  function overview(rows) {
    const s=summary(rows),ds=dsRows(rows),ts=Object.keys(TASKS),rids=new Set(rows.map(i=>i.r));
    let body=panel('三个后果域 · 八项任务','内圈为后果域，外圈为任务；扇区大小表示题数，点击查看完整 QA。',taxonomy(rows),['vsi','mmbench'],true);
    body+=panel('数据来源 × 任务','0 表示有候选但当前未入选；无供给表示对应 meta 任务表为空或不存在。',matrix(rows,ds,ts,i=>i.dataset,i=>i.task,d=>d.toUpperCase(),t=>TASKS[t],(d,t)=>![...rids].some(r=>DATA.releases[r].available_tasks?.[d]?.includes(t))),['robo','action'],true);
    body+=panel('题目、观测和图片分别计数','独立初始图使用未打点原图哈希；输入图包括实际展示的打点图与四张选项。',metrics([[fmt(s.observations),'独立观测'],[fmt(s.initial_images),'独立初始图'],[fmt(s.input_images),'独立输入图'],[fmt(s.image_references),'图片引用次数']])+bars(rows,i=>i.images.length,{order:[1,5],names:n=>`每题 ${n} 张图`}),['mmsi']);
    body+=panel('主集与半径诊断集','本表为加载的发布全量，独立于上方筛选；参数消融版本不重复计数。',table(['分区','题集','题数','观测数','对数'],Object.entries(DATA.statistics.sets).map(([k,s])=>[s.release,SETS[k.split('/')[1]],fmt(s.questions),fmt(s.observations),s.pairs?fmt(s.pairs):'—'])),['action','mmsi']);
    return section('overview','组成与规模','沿用项目既有分类；样本数和输入规模使用不同的计数单位。',body);
  }
  function coverage(rows) {
    const body=panel('场景覆盖与集中程度','分母为当前任务和来源的题数；meta 场景指采集成功场景，不代表全部均可出题。',table(['来源','入选场景','meta 成功场景','最大场景占比','前 5 场景占比'],dsRows(rows).map(d=>{const g=rows.filter(i=>i.dataset===d),sc=tally(g,i=>i.scene),rids=[...new Set(g.map(i=>i.r))];return [d.toUpperCase(),sc.length,rids.map(r=>DATA.releases[r].name+': '+DATA.releases[r].dataset_supply?.[d]?.captured_scenes).join(' / '),pct(sc[0]?.[1]||0,g.length),pct(sc.slice(0,5).reduce((s,x)=>s+x[1],0),g.length)];})),['robo','ec'],true)
      +panel('每个场景的题量分布','每个数据点是一个场景；各来源分别计算，不用合并总量掩盖集中程度。',datasetFacet(rows,g=>histogram(tally(g,i=>i.scene).map(([scene,n])=>({m:{n},scene,_items:g.filter(i=>i.scene===scene)})), 'n',[0,5,10,20,40,80,160,320,Infinity],'题')),['ec'],true)
      +panel('题量最多的场景','图示前 12 个场景，展开查看全部；同名场景按数据集区分。',bars(rows,i=>i.dataset.toUpperCase()+' / '+i.scene,{limit:12}),['ec'])
      +panel('各任务覆盖多少场景','按当前题目重新计算；同一场景在不同任务中出现，不可把各格直接求和。',table(['任务',...dsRows(rows).map(d=>d.toUpperCase())],Object.keys(TASKS).map(t=>[TASKS[t],...dsRows(rows).map(d=>new Set(rows.filter(i=>i.task===t&&i.dataset===d).map(i=>i.scene)).size)])),['robo'])
      +panel('每个观测承载的题数','半径对照两题共享一个观测；这里明确显示，不视为两张不同初始图。',bars(tally(rows,i=>i.dataset+'/'+i.obs).map(([obs,n])=>({obs,n,_items:rows.filter(i=>i.dataset+'/'+i.obs===obs)})),i=>i.n,{names:n=>`${n} 题 / 观测`}),['mmsi'])
      +panel('跨分区场景与输入图交集','本表使用两个发布主集的完整集合。相同文件内容按发布哈希识别；不是视觉近重复检验。',table(['发布组合','场景交集','输入图哈希交集'],DATA.statistics.overlaps.map(x=>[`${DATA.releases[x.a].name} / ${DATA.releases[x.b].name}`,x.scenes,x.images])),['clevr']);
    return section('coverage','场景与图像覆盖','分别看来源、任务和场景，避免用总题量代替覆盖程度。',body);
  }
  function actions(rows) {
    const reps=[['atomic','原子动作',COLORS[0]],['merged','合并动作',COLORS[1]],['coarse','粗粒度动作',COLORS[2]]],s=summary(rows);
    let body=panel('动作表达 × 任务','原子、合并和粗粒度是描述形式；不把该比例当成物理程序多样性。',stacked(rows,i=>i.task,i=>i.rep,reps,t=>TASKS[t]),['action']);
    body+=panel('动作表达 × 数据来源','每根条内部归一化，保留样本量；方便发现某个来源的粗粒度集中。',stacked(rows,i=>i.dataset,i=>i.rep,reps,d=>d.toUpperCase()),['action']);
    body+=panel('公开动作命令数','统计模型实际看到的命令数；合并写法可能减少命令数。',histogram(rows,'commands',[0,2,3,4,5,6,8,12,20,Infinity],'条'),['clevr']);
    body+=panel('累计前进距离','按整段公开动作相加，单位统一为米。此处表示计划动作，不是碰撞后实际走过的距离。',histogram(rows,'forward_m',[0,.25,.5,1,1.5,2,3,5,8,Infinity],'m'),['vsi','video']);
    body+=panel('累计转角与转向方向','累计转角取各段绝对值之和；净转角不能替代总转动量。',histogram(rows,'turn_abs_deg',[0,15,30,45,60,90,120,180,360,Infinity],'°')+bars(rows,i=>i.m.turn_side,{order:['无转向','左转','右转','双向']}),['video']);
    body+=panel('起手类型 × 任务','转向开头与前进开头的实际占比。',stacked(rows,i=>i.task,i=>i.m.start,[['turn','转向开头',COLORS[1]],['forward','前进开头',COLORS[0]]],t=>TASKS[t]),['gqa']);
    body+=panel('动作模式','连续同类命令合并为一个模式段；保留具体距离、转角的统计在其他图中。',bars(rows,i=>i.m.pattern,{limit:10}),['clevr']);
    const pg=tally(rows,i=>i.program),labels=new Map(rows.map(i=>[i.program,i.m.program_text]));
    body+=panel('高频程序与集中程度','按原子程序 ID 计数，原子／合并表达共享程序 ID；下方动作文字是该程序的一种表达。',metrics([[fmt(s.programs),'不同程序'],[pct(s.top10_program_questions,rows.length),'前 10 程序题量占比'],[pct(s.program_repeat_excess,rows.length),'超出首次出现的重复题量']])+bars(rows,i=>i.program,{limit:10,names:id=>labels.get(id)})+`<p class="distribution-note">共 ${fmt(pg.length)} 个程序；每个程序的首个样本也计入前十占比。</p>`,['clevr','gqa']);
    const relevant=rows.filter(i=>finite(i.m.prefix_commands));
    body+=panel('查询对应的动作前缀','首次碰撞、中途查询、终点任务的已记录前缀。是否碰撞与接触类别未记录的前缀不推测补齐。',histogram(relevant,'prefix_commands',[0,2,3,4,5,6,8,12,20,Infinity],'条')+`<p class="distribution-note">有记录 ${fmt(relevant.length)} / ${fmt(rows.length)}；该前缀不等于模型所需的最少推理步数。</p>`,['clevr']);
    body+=panel('查询前缀的前进距离','中途只累计查询比例对应的部分；首次碰撞只累计到接触点。',histogram(relevant,'prefix_forward_m',[0,.1,.25,.5,1,2,4,8,Infinity],'m'),['clevr','video']);
    return section('actions','动作覆盖与程序集中程度','将表达形式、物理运动和查询前缀分开。',body);
  }
  function parameters(rows) {
    let body=panel('相机高度 × 水平 FOV','每个数据集单独展示。数字为题数；未出现的组合显示 0，不推断原因。',datasetFacet(rows,g=>matrix(g,ordinal(new Set(g.map(i=>i.m.camera_height_m))),ordinal(new Set(g.map(i=>i.m.hfov_deg))),i=>i.m.camera_height_m,i=>i.m.hfov_deg,x=>x+' m',x=>x+'°')),['robo','action'],true);
    body+=panel('机器人半径 × 任务','按题面展示半径统计；空间关系与未来视图任务的合法半径不改变其目标答案。',matrix(rows,Object.keys(TASKS),ordinal(new Set(rows.map(i=>i.radius))),i=>i.task,i=>i.radius,t=>TASKS[t],r=>r+' m'),['gqa']);
    body+=panel('单位表达 × 任务','m/cm 分别计数；所有几何图统一用米，避免把单位转换误认为数值差异。',stacked(rows,i=>i.task,i=>i.unit,[['m','米',COLORS[0]],['cm','厘米',COLORS[2]]],t=>TASKS[t]),['video']);
    body+=panel('题面模板使用','每个任务内按模板序号计数；这是模板覆盖，不宣称自然语言多样性。',matrix(rows,Object.keys(TASKS),[...new Set(rows.map(i=>i.template.split('_').at(-1)))].sort(),i=>i.task,i=>i.template.split('_').at(-1),t=>TASKS[t],x=>'模板 '+x),['clevr']);
    body+=panel('题干长度','以空白分隔的词数，包含参数、动作及答案格式说明；不是推理难度评分。',histogram(rows,'prompt_words',[0,50,100,150,200,250,300,400,600,Infinity],'词'),['video','clevr']);
    return section('parameters','参数与语言表达','检查组合覆盖，显示来源和任务内部的分布。',body);
  }
  function collision(rows) {
    const yesno=taskRows(rows,'A1'),step=taskRows(rows,'A2'),category=taskRows(rows,'A3');let body='';
    if(yesno.length){
      body+=panel('碰撞率 · 来源与半径','只统计“是否碰撞”；半径诊断题若被选中，会按两条题目计数。50% 是参考线，不自动等同于发布硬规则。',conditional(yesno,i=>`${i.dataset.toUpperCase()} / ${i.radius} m`,'来源 / 半径'),['gqa']);
      body+=panel('碰撞率 · 长度与起手','按累计前进距离和起手类型交叉统计；每格同时显示分母。',conditional(yesno,i=>`${i.m.forward_m<.5?'<0.5':i.m.forward_m<1?'0.5–1':i.m.forward_m<2?'1–2':'≥2'} m / ${i.m.start==='turn'?'转向':'前进'}开头`,'长度 / 起手'),['gqa']);
    }
    if(step.length){
      body+=panel('首次碰撞所在的前进段','按题面前进命令总数分组。只有两次前进时不存在中间档。',matrix(step,ordinal(new Set(step.map(i=>i.m.forwards))),['首次','中间','末次'],i=>i.m.forwards,i=>i.m.position,n=>`${n} 次前进`),['gqa']);
      const groups=tally(step.filter(i=>i.m.longest_hit!==null),i=>i.dataset+'/'+i.m.forwards);
      body+=panel('是否命中唯一最长前进段','仅存在唯一最长段的题参与；并列最长不计入分母。参考为该组实际前进数的 1/F。',table(['来源 / 前进次数','命中 / 有效题数','命中率','1/F 参考'],groups.map(([k,n])=>{const g=step.filter(i=>i.dataset+'/'+i.m.forwards===k&&i.m.longest_hit!==null),hit=g.filter(i=>i.m.longest_hit).length;return [k.toUpperCase(),`${hit} / ${n}`,pct(hit,n),pct(1,g[0].m.forwards)];}))+`<p class="distribution-note">排除并列最长 ${fmt(step.filter(i=>i.m.longest_hit===null).length)} 题。</p>`,['gqa']);
      body+=panel('碰撞动作的公开编号','区分公开动作编号与第几次前进；转向命令也占公开编号。',bars(step,i=>i.m.collision_action,{order:ordinal(new Set(step.map(i=>i.m.collision_action))),names:n=>'第 '+n+' 个命令'}),['clevr']);
      body+=panel('接触点距该前进命令末端','边界附近题量的描述性统计；不改变碰撞归属或筛选门槛。',histogram(step,'command_end_margin_m',[0,.01,.025,.05,.1,.25,.5,1,Infinity],'m'),['gqa']);
    }
    if(category.length){body+=panel('首次接触类别的长尾','展示前 15 类并可展开全部。类别使用已发布答案名称，不在页面重新合并。',bars(category,i=>i.answer,{limit:15}),['gqa']);body+=panel('类别覆盖 × 来源','原始资产类别与公开类别可能有粒度差异；此表只统计公开类别。',table(['来源','类别数','仅 1 题的类别','前 10 类占比','最大类别占比'],dsRows(category).map(d=>{const g=category.filter(i=>i.dataset===d),c=tally(g,i=>i.answer);return [d.toUpperCase(),c.length,c.filter(x=>x[1]===1).length,pct(c.slice(0,10).reduce((s,x)=>s+x[1],0),g.length),pct(c[0][1],g.length)];})),['gqa','robo']);}
    return body?section('collision',DOMAINS.collision.name,'检查条件答案分布，避免总体配平掩盖任务内部偏差。',body):'';
  }
  function spatial(rows) {
    const distances=taskRows(rows,'B1','B3'),directions=taskRows(rows,'A4','B2'),checkpoints=taskRows(rows,'A4','B3');let body='';
    if(distances.length){
      body+=panel('查询时刻的目标距离','中途与终点可用任务筛选分别查看；物理量统一为米，展示分档固定便于跨版本比较。',histogram(distances,'distance_m',[0,1,2,4,8,16,Infinity],'m',COLORS[1]),['vsi']);
      body+=panel('初始目标距离','与查询距离使用相同分档。分布统计使用存储的连续 GT。',histogram(distances,'initial_distance_m',[0,1,2,4,8,16,Infinity],'m',COLORS[1]),['vsi']);
      body+=panel('带正负号的距离变化','查询距离减去初始距离；负值表示变近。绝对变化不能替代这项统计。',histogram(distances,'distance_delta_m',[-Infinity,-2,-1,-.3,-.05,.05,.3,1,2,Infinity],'m',COLORS[1]),['gqa']);
      body+=panel('距离变化类型 × 任务与来源','基本不变：<0.05 m；小变化：0.05–0.30 m；明显变化：≥0.30 m。',stacked(distances,i=>i.dataset.toUpperCase()+' / '+TASKS[i.task],i=>i.m.distance_change,[['明显变近','明显变近',COLORS[0]],['明显变远','明显变远',COLORS[2]],['小变化','小变化',COLORS[1]],['基本不变','基本不变','#9baba0']]),['gqa']);
    }
    for(const t of ['A4','B2']){const g=taskRows(directions,t);if(g.length)body+=panel(TASKS[t]+' · 水平 × 垂直标签','联合显示 24 类；水平八类和垂直三类不是独立的 24 种能力。',matrix(g,VERTICAL,HORIZONTAL,i=>i.m.vertical,i=>i.m.horizontal,k=>VN[VERTICAL.indexOf(k)],k=>HN[HORIZONTAL.indexOf(k)]),['vsi','mmsi']);}
    if(directions.length){
      body+=panel('方位角与仰角','连续角度来自 meta；同样数量的方向标签不代表相同几何角度覆盖。',histogram(directions,'azimuth_deg',[-180,-135,-90,-45,0,45,90,135,180],'°',COLORS[1])+histogram(directions,'elevation_deg',[-90,-45,-15,-7.5,7.5,15,45,90],'°',COLORS[1]),['vsi']);
      body+=panel('距离方向分界线的余量','小余量代表接近既定标签分界；图表不重新定义方向扇区。',histogram(directions,'boundary_margin_deg',[0,1,2,3,5,7.5,10,15,30,Infinity],'°',COLORS[1]),['gqa']);
    }
    if(checkpoints.length){
      body+=panel('中途查询的完成比例','同一百分比可以出现在不同前进命令，需和右侧动作位置一起看。',stacked(checkpoints,i=>i.task,i=>i.m.checkpoint_fraction,[[.25,'25%',COLORS[0]],[.5,'50%',COLORS[1]],[.75,'75%',COLORS[2]]],t=>TASKS[t]),['clevr']);
      body+=panel('中途查询发生在哪个命令','统计公开动作编号；不把完整动作列表长度当成查询前缀长度。',bars(checkpoints,i=>i.m.checkpoint_action,{order:ordinal(new Set(checkpoints.map(i=>i.m.checkpoint_action))),names:n=>'第 '+n+' 个命令'}),['clevr']);
    }
    return body?section('spatial',DOMAINS.spatial.name,'联合展示连续几何量、离散答案和查询时刻。',body):'';
  }
  function visual(rows) {
    const g=taskRows(rows,'C1');if(!g.length)return '';
    const options=unique(g.flatMap(i=>i.m.option_quality.map((q,n)=>({...i,image_hash:i.image_hashes[n+1],m:{dark:q[0],edges:q[1]}}))),i=>i.image_hash);
    const body=panel('正确选项位置 × 来源','统计实际发布排列的标准答案；没有运行模型，不展示模型位置偏好。',matrix(g,dsRows(g),['A','B','C','D'],i=>i.dataset,i=>i.answer,d=>d.toUpperCase()),['mmbench'])
      +panel('选项之间的最小图像差异','使用采集时记录的 min_block_l1；只描述分布，不据此去重、替换选项或宣称难度。',histogram(g,'min_block_l1',[0,.01,.02,.04,.08,.12,.2,.4,Infinity],'L1',COLORS[2]),['action'])
      +panel('选项图的暗像素比例','按独立选项图哈希计数，一张图只算一次；初始图另列在画质部分。',histogram(options,'dark',[0,.005,.01,.02,.03,.04,.05,.1,1],'比例',COLORS[2]),['spar'])
      +panel('选项图的边缘密度','低纹理与贴近墙面本身不判为坏图；此图只用于描述输入图像。',histogram(options,'edges',[0,.01,.025,.05,.1,.2,.3,.5,1],'比例',COLORS[2]),['spar']);
    return section('visual',DOMAINS.visual.name,'展示实际候选组的组成与图像指标，保持发布选项不变。',body);
  }
  function diagnostic(rows) {
    const pairs=rows.filter(i=>i.set==='radius_pairs');if(!pairs.length)return '';
    const groups=new Map();for(const i of pairs){if(!groups.has(i.pair))groups.set(i.pair,[]);groups.get(i.pair).push(i);}
    const entries=[...groups.values()].map(g=>({group:g,dataset:g[0].dataset,combo:g.map(i=>i.radius).sort((a,b)=>a-b).join(' / '),unit:g[0].unit,complete:g.length===2}));
    const complete=entries.filter(e=>e.complete),incomplete=entries.length-complete.length;
    let body=panel('诊断集按对计数','当前筛选可能只留下半对；不完整对单列，不参与配对分布。',metrics([[fmt(pairs.length),'题目'],[fmt(complete.length),'完整对'],[fmt(incomplete),'筛选后不完整对']])+table(['来源','完整对数','米制对','厘米制对'],dsRows(pairs).map(d=>{const g=complete.filter(e=>e.dataset===d);return [d.toUpperCase(),g.length,g.filter(e=>e.unit==='m').length,g.filter(e=>e.unit==='cm').length];})),['action']);
    body+=panel('半径组合覆盖','每对只计一次；诊断题不并入主 benchmark 总量。',bars(complete.map(e=>({...e.group[0],combo:e.combo,_items:e.group})),i=>i.combo,{names:x=>x+' m'}),['gqa']);
    return section('pairs','半径敏感性诊断','同一画面与动作、不同半径的配对设置单独报告。',body);
  }
  function quality(rows) {
    const uniqueImages=unique(rows,i=>i.raw_hash),collision=taskRows(rows,'A1'),evidence=taskRows(rows,'A1','A2'),visible=collision.filter(i=>i.m.corridor_samples>0),cats=taskRows(rows,'A3');
    let body=panel('初始图暗像素比例','按未打点原图哈希去重计数；指标定义为三个 RGB 通道均小于 12 的像素比例。',histogram(uniqueImages,'dark',[0,.005,.01,.02,.03,.04,.05,.1,1],'比例'),['spar']);
    const edgeMeasured=uniqueImages.filter(i=>Number.isFinite(i.m.edges)),lowEdge=edgeMeasured.filter(i=>i.m.edges<.01).length;
    body+=panel('初始图边缘密度与近乎空白画面','边缘密度 < 0.01 的原图单独计数，作为人工复核线索；正常白墙或贴脸仍保留，不作为淘汰条件。',metrics([[fmt(lowEdge),'低边缘原图'],[fmt(edgeMeasured.length),'已测量原图'],[edgeMeasured.length?(100*lowEdge/edgeMeasured.length).toFixed(2)+'%':'—','低边缘占比']])+histogram(uniqueImages,'edges',[0,.01,.025,.05,.1,.2,.3,.5,1],'比例'),['spar']);
    body+=panel('有效深度与可见地面','观测级元数据，不是可回答性的通过证书；每个独立初始图只计一次。',histogram(uniqueImages,'valid_depth_ratio',[0,.2,.4,.6,.8,.9,.95,1],'有效深度比例')+histogram(uniqueImages,'visible_floor_ratio',[0,.01,.05,.1,.2,.4,.6,1],'地面比例'),['spar']);
    body+=panel('深度证据与 GT 的一致程度','深度推断与精确 GT 是不同方法；一致率只作辅助证据，不能自动作为题目正确率。',evidence.length?stacked(evidence,i=>i.dataset.toUpperCase()+' / '+TASKS[i.task],i=>i.m.depth_agree,[[true,'一致',COLORS[0]],[false,'不一致',COLORS[2]]]):empty(),['gqa']);
    body+=panel('路径覆盖与未知证据','路径采样数为 0 时记为未知，不把其覆盖率当作 0 或 100%。',metrics([[fmt(collision.length-visible.length),'路径零采样题'],[fmt(visible.length),'有路径采样题']])+histogram(visible,'corridor_coverage',[0,.2,.4,.6,.8,.9,1],'覆盖率'),['spar']);
    body+=panel('接触区可见点数','首次接触类别任务的认证证据；点数多不等同于人类可答。',histogram(cats,'visible_points',[0,10,20,50,100,250,500,1000,3000,Infinity],'点'),['spar']);
    body+=panel('人工审阅记录','记录保存在本机浏览器；定向抽查样本不能直接用于估计整库错误率。',bars(rows,status,{order:['未记录','有记录','有问题']}),['spar']);
    return section('quality','图像与可回答性证据','区分客观图像指标、几何辅助证据和人工判断。',body);
  }
  function baselines(rows) {
    const entries=[];
    for(const t of Object.keys(TASKS)){
      const g=taskRows(rows,t);if(!g.length)continue;
      let chance='不定义',freq='—',other='—';
      if(!['B1','B3'].includes(t)){freq=pct(tally(g,i=>i.answer)[0][1],g.length);if(t==='A1')chance='50.0%';if(['A4','B2'].includes(t))chance=pct(1,24);if(t==='C1')chance='25.0%';if(t==='A2'){chance=pct(mean(g.map(i=>1/i.m.commands)),1);other='只猜前进：'+pct(mean(g.map(i=>1/i.m.forwards)),1);}}
      else {other=[.25,.5].map(tol=>`${tol} m：${pct(g.filter(i=>Math.abs(i.m.initial_distance_m-i.m.distance_answer_m)<=tol+1e-12).length,g.length)}`).join(' / ');}
      entries.push([TASKS[t],g.length,chance,freq,other]);
    }
    return section('baselines','答案先验与诊断参考','数据分布基线与模型成绩分开；当前不生成或推测模型性能。',panel('任务适用的参考基线','频次是本筛选子集的多数答案占比，不是训练所得基线。距离列使用 GT 初始距离，仅为诊断；对照公开答案按既定容差计分。',table(['任务','题数','均匀随机','多数答案占比','结构 / 初始距离诊断'],entries)+`<p class="distribution-note">无图模型、人类作答、固定距离常数基线：尚未在此统计流程运行。改变筛选范围会改变多数类占比。</p>`,['clevr','open'],true));
  }
  function compare() {
    const rows=DATA.items.filter(i=>matches(i,'release'));
    const reps=[['atomic','原子动作',COLORS[0]],['merged','合并动作',COLORS[1]],['coarse','粗粒度动作',COLORS[2]]];
    const rids=[...new Set(rows.map(i=>i.r))];if(rids.length<2)return '';
    const columns=rids.map(r=>DATA.releases[r].name),features=[['commands','公开命令数','条'],['forward_m','累计前进距离','m'],['turn_abs_deg','累计转角','°'],['distance_m','查询目标距离','m'],['boundary_margin_deg','方向边界余量','°']];
    const body=panel('分区间的连续量比较','除评测分区外沿用当前筛选；每格显示中位数 [P10, P90] 和有效样本数。',table(['指标',...columns],features.map(([key,name,unit])=>[name,...rids.map(r=>{const v=numeric(rows.filter(i=>i.r===r),key);return v.length?`${quantile(v,.5).toFixed(2)} [${quantile(v,.1).toFixed(2)}, ${quantile(v,.9).toFixed(2)}] ${unit} · n=${v.length}`:'不适用';})])),['video'],true)
      +panel('分区间的动作表达','每根条内部使用百分比，避免主集大小不同造成视觉误判。',stacked(rows,i=>DATA.releases[i.r].name+' / '+i.dataset.toUpperCase(),i=>i.rep,reps),['action'])
      +panel('分区间的距离变化','在距离任务内比较；显示原始样本数，反映分布差异而不自动判定错误。',stacked(taskRows(rows,'B1','B3'),i=>DATA.releases[i.r].name+' / '+i.dataset.toUpperCase(),i=>i.m.distance_change,[['明显变近','明显变近',COLORS[0]],['明显变远','明显变远',COLORS[2]],['小变化','小变化',COLORS[1]],['基本不变','基本不变','#9baba0']]),['gqa']);
    return section('compare','Seen / unseen 对照','使用相同物理单位、统计定义和筛选条件。',body);
  }
  function provenance(rows) {
    const rids=[...new Set(rows.map(i=>i.r))],supply=[];
    for(const r of rids)for(const[k,v]of Object.entries(DATA.releases[r].report?.candidates||{})){const[d,t]=k.split('/');if(!rows.some(i=>i.r===r&&i.dataset===d&&i.task===t))continue;supply.push([DATA.releases[r].name,d.toUpperCase(),TASKS[t],fmt(v.rows),fmt(v.kept),fmt(v.observations)]);}
    let body=panel('构建时的候选供给','原始构建报告的全量记录：包含训练候选，不是当前筛选的拒绝率；不同格子可能共享观测，不直接相加。',table(['分区','来源','任务','压缩前行数','压缩后行数','候选观测'],supply),['spar'],true);
    const eligible=[];
    for(const r of rids)for(const[k,v]of Object.entries(DATA.releases[r].report?.benchmark_supply||{})){
      const[d,t,...cell]=k.split('/');if(!TASKS[t]||!rows.some(i=>i.r===r&&i.dataset===d&&i.task===t))continue;
      let name=cell.join(' / ');
      if(t==='A1')name=cell[0]==='True'?'碰撞':'不碰撞';
      else if(['A4','B2'].includes(t)&&/^\d+$/.test(name)){const n=Number(name);name=HN[Math.floor(n/3)]+' / '+VN[n%3];}
      else name=name.replace('F3+','≥3 次前进').replace('F2','2 次前进').replace('first','首次').replace('middle','中间').replace('last','末次').replace('all','全部');
      eligible.push({r,d,t,name,before:v[0],after:v[1]});
    }
    eligible.sort((a,b)=>a.after-b.after);
    const supplyRows=eligible.map(e=>[DATA.releases[e.r].name,e.d.toUpperCase(),TASKS[e.t],e.name,fmt(e.before),fmt(e.after),pct(e.after,e.before)]);
    const head=['分区','来源','任务','条件格','图像检查前','检查后','保留率'];
    body+=panel('图像筛选前后 · 条件格供给','构建报告中的独立候选观测数，展示检查后供给最少的 20 格。它们可以共享观测，不相加，也不把压缩后行数当作独立题数。',table(head,supplyRows.slice(0,20))+(supplyRows.length>20?`<details class="chart-details"><summary>展开全部 ${supplyRows.length} 格</summary>${table(head,supplyRows)}</details>`:''),['spar','gqa'],true);
    body+=panel('统计来源与重算','读取发布 QA、选题清单、构建报告和对应 meta；没有读取可变的当前选题 spec。',table(['分区','题面版本','发布哈希'],rids.map(r=>[DATA.releases[r].name,DATA.releases[r].prompt_version,DATA.releases[r].release_sha256]))+`<p class="distribution-note">分档柱图的横轴是离散区间，柱高为样本数，不是概率密度；区间左闭右开，有限的最末端包含上界。更换发布目录后重新运行页面生成命令即可更新统计；源码和输入清单摘要见 <a href="stats.json" download>stats.json</a>。同一发布的浏览器审阅记录继续保留。</p>`,[],true);
    body+=panel('参考方法及本项目适配','来源说明统计方法的依据。任务语义、物理动作、机器人参数和查询前缀是本项目字段，并非这些论文的原始指标。',`<div class="method-list">${Object.values(SOURCES).map(s=>`<div><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.name)}</a><p>${esc(s.method)}</p></div>`).join('')}</div>`,[],true);
    return section('methods','供给、口径与来源','所有图表由发布数据重算，缺失或不适用的项目明确标注。',body);
  }
  function render(rows) {
    if(!DATA.statistics||!rows.every(i=>i.m)){$('stats-content').innerHTML=empty('此页面缓存尚无完整统计字段，请重新运行 scripts.review_page。');return;}
    drills=[];panels=0;
    if(!rows.length){$('stats-content').innerHTML=empty('当前筛选没有题目，请重置筛选。');return;}
    const s=summary(rows),allMain=DATA.items.filter(i=>i.set==='benchmark').length,allPairs=new Set(DATA.items.filter(i=>i.set==='radius_pairs').map(i=>i.pair)).size;
    let content=overview(rows)+coverage(rows)+actions(rows)+parameters(rows)+collision(rows)+spatial(rows)+visual(rows)+diagnostic(rows)+quality(rows)+baselines(rows)+compare()+provenance(rows);
    const groups=[['overview','组成'],['coverage','场景 / 图像'],['actions','动作'],['parameters','参数 / 题面'],['collision','碰撞事件'],['spatial','空间关系'],['visual','视觉后果'],['pairs','半径诊断'],['quality','质量证据'],['baselines','参考基线'],['compare','分区对照'],['methods','方法 / 来源']];
    const nav=groups.filter(([k])=>content.includes(`id="stat-${k}"`)).map(([k,n])=>`<a href="#stat-${k}">${n}</a>`).join('');
    $('stats-content').innerHTML=`<div id="stats-top" class="stats-intro"><div><div class="eyebrow">DATASET ANATOMY</div><h2>从组成，到覆盖与偏差。</h2><p>${panels} 组统计图表 · 点击图中样本组查看完整 QA · 当前 ${fmt(rows.length)} 题</p></div><button class="quiet-button" id="download-statistics">导出当前统计 JSON ↓</button></div><div class="release-scope-note">已加载主集共 <b>${fmt(allMain)}</b> 题；半径诊断另有 <b>${fmt(allPairs)}</b> 对。下面图表跟随筛选，标注“发布全量”的表除外。</div>${metrics([[fmt(s.questions),'当前题目'],[fmt(s.scenes),'入选场景'],[fmt(s.observations),'独立观测'],[fmt(s.initial_images),'独立初始图'],[fmt(s.programs),'不同程序'],[fmt(s.pairs),'诊断对 ID']])}<nav class="stats-nav" aria-label="统计目录">${nav}</nav>${content}`;
    $('download-statistics').onclick=()=>{
      const data={schema:'pbench.filtered-statistics.v1',summary:summary(rows),scope:$('scope').textContent,
        releases:DATA.releases.map(r=>({name:r.name,release_sha256:r.release_sha256})),
        items:rows.map(i=>({id:i.id,release:DATA.releases[i.r].name,set:i.set,task:TASKS[i.task],domain:DOMAINS[i.domain].name,dataset:i.dataset,scene:i.scene,obs:i.obs,program:i.program,representation:i.rep,unit:i.unit,radius_m:i.radius,answer:i.gt,measurements:i.m})),
        sources:SOURCES,definitions:DATA.statistics.notes};
      const u=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='pbench-statistics.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);
    };
  }
  return {render,summary,measurements:rows=>rows.map(i=>i.m),drill:index=>drills[index],quantile};
})();
