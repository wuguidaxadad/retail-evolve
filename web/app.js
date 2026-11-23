const $ = (id) => document.getElementById(id);
const escape = (s) => String(s ?? "").replace(/[&<>"']/g, (x) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[x]));
let state = {jobs:[], versions:[], overview:null, conversation:sessionStorage.getItem("retail-conversation"), chatBusy:false, project:null, cases:[], selectedCase:null};
let renderedVersions = "";
const names = {evaluate:"业务评测",learn:"ACE 经验学习",evolve:"历史受限优化",rsi:"完整 RSI 自进化",gate:"候选检查",chat:"客服会话"};
const statuses = {queued:"等待执行",running:"正在执行",succeeded:"已完成",failed:"失败",cancelled:"已取消",candidate:"待检查",validated:"已通过",rejected:"未通过",invalidated:"开发记录 · 已作废",superseded:"旧注入方式 · 已替换"};
async function api(path, options={}) {
  let response;
  try {
    response = await fetch("/api" + path, {...options, headers:{"Content-Type":"application/json", ...options.headers}});
  } catch {
    throw Error("暂时无法连接本地服务，请稍后重试。");
  }
  const data = await response.json();
  if (!response.ok) throw Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  return data;
}
function toast(text) {$("toast").textContent=text; $("toast").classList.remove("hidden"); setTimeout(()=>$("toast").classList.add("hidden"),5500);}
function navigate(page) {
  history.replaceState(null, "", "#"+page);
  document.querySelectorAll(".page").forEach(el=>el.classList.toggle("hidden",el.id!==page));
  document.querySelectorAll("nav button").forEach(el=>el.classList.toggle("selected",el.dataset.page===page));
  $("breadcrumb").textContent=document.querySelector('nav button[data-page="'+page+'"]').textContent;
}
document.querySelectorAll("nav button").forEach(el=>el.onclick=()=>navigate(el.dataset.page));
function stat(label,value,note) {return '<div class="stat"><span>'+escape(label)+'</span><b>'+escape(value)+'</b><span>'+escape(note)+'</span></div>';}
function resultText(job) {
  if(job.status==="invalidated") return "早期开发适配有误，已作废；请使用后续有效结果。";
  if (!job.result) return job.error ? job.error.slice(0,140) : job.stage;
  if (job.kind==="evaluate" && typeof job.result.mean_reward==="number") return '平均得分 '+job.result.mean_reward.toFixed(2)+' · '+job.result.sample_count+' 题 × '+job.result.trials+' 次';
  if (job.kind==="gate") return (job.result.passed?"检查通过":"检查未通过")+' · '+job.result.incumbent_score.toFixed(2)+' → '+job.result.candidate_score.toFixed(2);
  return job.result.version_id ? '候选 '+job.result.version_id.slice(0,8) : '客服已回复';
}
function jobTable(jobs) {
  if (!jobs.length) return '<p class="muted">暂无任务，提交一次业务评测即可开始。</p>';
  return '<table><thead><tr><th>任务</th><th>状态</th><th>进度 / 结果</th><th>操作</th></tr></thead><tbody>'+jobs.map(j=>
    '<tr><td>'+escape(names[j.kind])+'<div class="small">'+escape(j.id.slice(0,8))+' · '+escape(j.payload.split||'')+'</div><div class="small">'+escape((j.payload.model_config||j.result?.model_config||{}).agent||'')+'</div></td><td><span class="pill '+(j.status==="failed"?"bad":"")+'">'+escape(statuses[j.status])+'</span><div class="small">执行 '+j.attempts+' 次</div></td><td>'+escape(resultText(j))+
    (j.status==="running"?'<div class="progress"><i style="width:'+Math.round(j.progress*100)+'%"></i></div>':'')+'</td><td>'+
    (["evaluate","gate"].includes(j.kind)?'<button onclick="showTraces(\''+j.id+'\')">轨迹</button>':'')+
    (["queued","running"].includes(j.status)?'<button onclick="cancel(\''+j.id+'\')">取消</button>':'')+'</td></tr>').join("")+'</tbody></table>';
}
function preserveOptions(element, options) {
  const old=element.value; element.innerHTML=options;
  if ([...element.options].some(x=>x.value===old)) element.value=old;
}
function renderVersions() {
  const active=state.overview.deployment.version_id;
  const currentIds=new Set([active,...Object.values(state.project?.versions||{})]);
  const visible=state.versions.filter(v=>$("version-scope").value==="all"||currentIds.has(v.id));
  $("version-list").innerHTML=visible.map(v=>{
    const book=v.skillbook.skills || {};
    const skills=Array.isArray(book)?book:Object.values(book);
    return '<article class="card"><div class="row"><h2>'+escape(v.name)+(active===v.id?' <span class="pill">当前服务版本</span>':'')+'</h2><span class="pill '+(v.status==="rejected"?"bad":"")+'">'+escape(statuses[v.status])+'</span></div><p class="muted">'+escape(v.origin)+' · '+escape(v.id)+' · 父版本 '+escape(v.parent_id||'无')+'</p>'+
    (v.status==="validated"&&active!==v.id?'<button class="primary" onclick="publish(\''+v.id+'\')">发布此版本</button> ':'')+
    (v.metadata.baseline&&active!==v.id?'<button onclick="rollback(\''+v.id+'\')">回滚至原始版本</button>':'')+
    '<h3>业务经验 / 执行经验 · '+skills.length+' 条</h3>'+
    (skills.length?skills.map(s=>'<div class="skill"><b>'+escape(s.issue)+'</b><p>'+escape(s.insight)+'</p><div class="small">'+escape(s.section)+' · '+escape(s.id)+' · 有益 '+escape(s.helpful_count)+' / 有害 '+escape(s.harmful_count)+'</div></div>').join(""):'<p class="muted">尚未积累经验。</p>')+
    '<details><summary>执行策略代码</summary><pre>'+escape(v.harness_code)+'</pre></details>'+
    (v.metadata.code_diff?'<details><summary>与父版本的策略差异</summary><pre>'+escape(v.metadata.code_diff)+'</pre></details>':'')+
    '<details><summary>学习来源与检查记录</summary><pre>'+escape(JSON.stringify(v.metadata,null,2))+'</pre></details></article>';
  }).join("");
}
function render() {
  const o=state.overview;
  $("stats").innerHTML=stat("公开客户",o.records.users||0,"可用于客服身份与订单查询")+stat("公开订单",o.records.orders||0,"涵盖修改、取消与退换货")+stat("公开商品",o.records.products||0,"包含不同规格与库存")+stat("业务任务",Object.values(o.tasks).reduce((a,b)=>a+b,0),"分组隔离，避免评测泄漏");
  $("split-stats").innerHTML='<span class="pill">学习 '+o.tasks.learn+'</span> <span class="pill">验证 '+o.tasks.validation+'</span> <span class="pill">最终保留 '+o.tasks.final+'</span>';
  $("recent-jobs").innerHTML=jobTable(state.jobs.slice(0,5));
  $("experiment-jobs").innerHTML=jobTable(state.jobs.filter(j=>j.kind!=="chat"));
  const versionOptions=state.versions.filter(v=>!["invalidated","superseded"].includes(v.status)).map(v=>'<option value="'+v.id+'">'+escape(v.name)+' · '+escape(statuses[v.status])+'</option>').join("");
  document.querySelectorAll(".version-select").forEach(el=>preserveOptions(el,versionOptions));
  const nativeOptions=state.versions.filter(v=>v.metadata.runtime_kind==="auto_harness_full"&&!["invalidated","superseded"].includes(v.status)).sort((a,b)=>(b.id===o.deployment.version_id)-(a.id===o.deployment.version_id)).map(v=>'<option value="'+v.id+'">'+escape(v.name)+'</option>').join("");
  document.querySelectorAll("#rsi-form .version-select").forEach(el=>preserveOptions(el,nativeOptions));
  const sources=state.jobs.filter(j=>j.kind==="evaluate"&&j.status==="succeeded"&&j.payload.split==="learn");
  preserveOptions($("learning-source"),sources.length?sources.map(j=>'<option value="'+j.id+'">'+escape(j.id.slice(0,8))+' · '+escape(j.payload.version_id.slice(0,8))+' · 得分 '+j.result.mean_reward.toFixed(2)+'</option>').join(""):'<option value="">请先完成学习集评测</option>');
  preserveOptions($("candidate-select"),state.versions.filter(v=>v.parent_id&&["candidate","rejected"].includes(v.status)).map(v=>'<option value="'+v.id+'">'+escape(v.name)+'</option>').join(""));
  const signature=JSON.stringify([state.versions,state.overview.deployment]);
  if(signature!==renderedVersions){renderVersions();renderedVersions=signature;}
}
async function refresh() {
  try {
    [state.overview,state.jobs,state.versions]=await Promise.all([api("/overview"),api("/jobs"),api("/versions")]);
    $("connection").textContent="本地服务运行中"; $("connection").classList.remove("bad");
    render();
    if(state.chatBusy) {
      const j=await api("/jobs/"+state.chatBusy);
      if(j&&["succeeded","failed","cancelled"].includes(j.status)) {
        state.chatBusy=false; $("chat-send").disabled=false;
        if(j.status!=="succeeded") toast(j.error||"任务已取消");
        await loadConversation();
      }
    }
  } catch(e) {$("connection").textContent="连接中断 · 正在重试"; $("connection").classList.add("bad");}
}
async function submit(form,kind) {
  const p=Object.fromEntries(new FormData(form));
  for(const k of ["limit","trials"]) if(k in p) p[k]=Number(p[k]);
  if(kind==="source") {
    const source=state.jobs.find(j=>j.id===p.source_job_id);
    if(!source) throw Error("请先完成学习集评测");
    p.version_id=source.payload.version_id;
  } else p.kind=kind;
  if(!p.version_id) throw Error("请选择版本");
  const job=await api("/jobs",{method:"POST",headers:{"Idempotency-Key":crypto.randomUUID()},body:JSON.stringify(p)});
  toast("任务已提交："+job.id.slice(0,8)); await refresh();
}
for(const [id,kind] of [["eval-form","evaluate"],["learn-form","source"],["gate-form","gate"],["rsi-form","rsi"]]) {
  $(id).onsubmit=async e=>{e.preventDefault(); const button=e.target.querySelector("button");button.disabled=true;try{await submit(e.target,kind);}catch(err){toast(err.message);}finally{button.disabled=false;}};
}
async function cancel(id) {try{await api("/jobs/"+id+"/cancel",{method:"POST"});await refresh();}catch(e){toast(e.message);}}
async function publish(id) {try{await api("/releases",{method:"POST",body:JSON.stringify({version_id:id,expected_revision:state.overview.deployment.revision})});toast("版本已发布，新会话将使用此版本");await refresh();}catch(e){toast(e.message);}}
async function rollback(id) {try{await api("/rollback",{method:"POST",body:JSON.stringify({version_id:id,expected_revision:state.overview.deployment.revision})});toast("已回滚");await refresh();}catch(e){toast(e.message);}}
async function showTraces(id) {
  navigate("experiments"); $("trace-card").classList.remove("hidden"); $("trace-data").textContent="正在加载";
  try {
    const traces=await api("/jobs/"+id+"/traces");
    $("trace-data").innerHTML=traces.length?traces.map(t=>'<h3>任务 '+escape(t.task_id)+' · '+escape(t.label)+' · 得分 '+t.reward+'</h3><p class="muted">'+escape(t.data.task_summary)+'</p>'+
      (t.data.messages||[]).map(m=>'<div class="trace-turn"><b>'+escape(m.role)+'</b><br>'+escape(m.content||JSON.stringify(m.tool_calls))+'</div>').join("")+
      '<details><summary>评测依据</summary><pre>'+escape(JSON.stringify(t.data.reward_info,null,2))+'</pre></details>').join(""):'<p class="muted">任务尚未完成一个样本。</p>';
  } catch(e){toast(e.message);}
}
$("new-chat").onclick=async()=>{try{const c=await api("/conversations",{method:"POST"});state.conversation=c.id;sessionStorage.setItem("retail-conversation",c.id);state.chatBusy=false;$("chat-send").disabled=false;await loadConversation();}catch(e){toast(e.message);}};
async function loadConversation(){
  if(!state.conversation)return;
  const c=await api("/conversations/"+state.conversation);
  $("chat-version").textContent="固定版本 "+c.version_id.slice(0,12)+" · 会话 "+c.id.slice(0,8);
  $("messages").innerHTML=c.messages.filter(m=>["assistant","user"].includes(m.role)&&m.content).map(m=>'<div class="bubble '+escape(m.role)+'">'+escape(m.content)+'</div>').join("")||'<p class="muted">输入订单问题，开始对话。</p>';
  $("messages").scrollTop=$("messages").scrollHeight;
  if(c.active_job){state.chatBusy=c.active_job;$("chat-send").disabled=true;}
  await loadOrder();
  $("tool-log").innerHTML=c.messages.filter(m=>m.role==="assistant"&&m.tool_calls).flatMap(m=>m.tool_calls).map(t=>'<div>'+escape(t.name)+'<br>'+escape(JSON.stringify(t.arguments))+'</div><br>').join("")||"尚无工具调用";
}
$("chat-form").onsubmit=async e=>{
  e.preventDefault();const message=$("chat-text").value.trim();if(!message||state.chatBusy)return;
  try{
    if(!state.conversation){const c=await api("/conversations",{method:"POST"});state.conversation=c.id;sessionStorage.setItem("retail-conversation",c.id);}
    const j=await api("/conversations/"+state.conversation+"/messages",{method:"POST",headers:{"Idempotency-Key":crypto.randomUUID()},body:JSON.stringify({message})});
    state.chatBusy=j.id;$("chat-send").disabled=true;$("chat-text").value="";
    $("messages").innerHTML+='<div class="bubble user">'+escape(message)+'</div><p class="muted">客服正在查询和处理…</p>';
  }catch(err){toast(err.message);}
};
async function operations(){
  try{const h=await api("/health");$("current-model").textContent=h.model;const o=await api("/operations");$("ops-stats").innerHTML=stat("等待投递",o.outbox_pending,"MySQL 待发送事件")+stat("队列消息",o.queue_length,"包含已确认的历史消息")+stat("未确认消息",o.pending.pending,"执行中或等待恢复")+stat("死信消息",o.dead_letters,"超限失败，保留供排查");
    $("releases").innerHTML=o.releases.map(r=>'<p>'+escape(r.action==="rollback"?"回滚":"发布")+' → '+escape(r.version_id.slice(0,8))+'<br><span class="small">'+escape(r.created_at)+'</span></p>').join("")||'<p class="muted">尚未发布新版本。</p>';
  }catch(e){$("ops-stats").textContent=e.message;}
}
async function init(){
  const page=location.hash.slice(1);
  if([...document.querySelectorAll("nav button")].some(b=>b.dataset.page===page))navigate(page);
  await refresh();
  try {
    const h=await api("/health");$("connection").textContent=Object.values(h.services).every(Boolean)?"本地服务运行中":"部分服务异常";
    state.cases=await api("/demo-cases"); renderCases();
    await loadProject();
    if(state.conversation) await loadConversation();
  }catch(e){toast(e.message);}
  await operations();setInterval(refresh,5000);setInterval(operations,15000);setInterval(loadProject,15000);
}


function renderCases(){
  $("business-cases").innerHTML=state.cases.map((c,i)=>'<article class="case-card"><span class="case-number">0'+(i+1)+' / AFTER SALES</span><h2>'+escape(c.title)+'</h2><p>'+escape(c.goal)+'</p><p class="small">'+escape(c.customer)+' · '+escape(c.order_id)+'</p><button class="primary" data-case="'+escape(c.id)+'">体验这个案例 →</button></article>').join('');
  $("examples").innerHTML=state.cases.map(c=>'<button class="example" data-case="'+escape(c.id)+'">'+escape(c.title)+'<br>'+escape(c.order_id)+'</button>').join('');
  document.querySelectorAll('[data-case]').forEach(b=>b.onclick=()=>selectCase(b.dataset.case));
}
async function selectCase(id){
  const c=state.cases.find(c=>c.id===id);if(!c)return;
  if (typeof createServiceTicket === 'function') {
    await createServiceTicket(c.order_id,{disabled:false});
    const input=$('service-message'); if(input) input.value=c.message;
    return;
  }
  try{
    if(state.chatBusy)throw Error("当前客服任务尚未结束，请等待或在任务列表取消后再切换案例。");
    const conv=await api('/conversations',{method:'POST'});
    state.selectedCase=c;state.conversation=conv.id;sessionStorage.setItem('retail-conversation',conv.id);sessionStorage.setItem('retail-case',c.id);
    navigate('chat');await loadConversation();$('chat-text').value=c.message;$('chat-text').focus();
  }catch(e){toast(e.message);}
}
async function loadOrder(){
  const c=state.selectedCase||state.cases.find(c=>c.id===sessionStorage.getItem('retail-case'));
  if(!c||!state.conversation)return;
  try{
    const data=await api('/conversations/'+state.conversation+'/orders/'+encodeURIComponent(c.order_id));
    const statusNames={pending:'待处理',delivered:'已送达',canceled:'已取消','return requested':'已申请退货','exchange requested':'已申请换货'};
    $('order-panel').innerHTML='<div class="order-state"><h3>当前会话的订单</h3><b>'+escape(c.order_id)+' · '+escape(statusNames[data.order.status]||data.order.status)+'</b><p>'+escape(data.order.items.map(i=>i.name).join('、'))+'</p><p>'+ (Object.keys(data.changes).length?'已记录业务变化：'+escape(Object.keys(data.changes).join('、')):'尚未修改订单')+'</p><details><summary>核对订单变化</summary><pre>'+escape(JSON.stringify(data.changes,null,2))+'</pre></details></div>';
  }catch(e){$('order-panel').textContent=e.message;}
}
async function loadProject(){
 try{
  const p=await api('/project');state.project=p;
  const selectedIds=new Set(Object.values(p.versions));
  preserveOptions($('service-version'),'<option value="">当前服务版本</option>'+state.versions.filter(v=>selectedIds.has(v.id)).map(v=>'<option value="'+v.id+'">'+escape(v.name)+'</option>').join(''));
  $('current-model').textContent='客服 '+p.models.customer;
  const labels={baseline:'原始 Agent',harness:'仅 Agent 自进化',skills:'仅 Skill 自进化'};
  $('comparison').innerHTML='<div class="compare-grid">'+Object.entries(p.comparison.arms).map(([k,b])=>'<div class="compare-arm"><h3>'+labels[k]+'</h3><b>'+b.completed+' / '+b.planned+'</b><span>有效会话已完成</span><p>'+ (b.success_rate===null?'成绩待完整评测后公布':'成功率 '+(b.success_rate*100).toFixed(1)+'%')+'</p><span>运行 '+(b.jobs.running||0)+' · 排队 '+(b.jobs.queued||0)+' · 异常 '+((b.jobs.failed||0)+b.invalid)+'</span></div>').join('')+'</div><p class="muted">学习 '+p.splits.learn+' 题 · 门禁 '+p.splits.validation+' 题 · 最终 '+p.splits.final+' 题。只纳入固定的 trial 0 记录；调用异常不计业务失败。</p>';
  $('rsi-history').innerHTML=(p.early_stop?'<p class="note">已按用户决定提前停止，固定通过门禁的最佳代码。门禁成绩不是最终测试成绩。</p>':'')+p.iterations.map(r=>'<div class="timeline-item"><b>第 '+r.iteration+' 轮</b> <span class="pill '+(r.kept?'':'bad')+'">'+(r.kept?'保留':'未保留')+'</span><p>修改范围：'+escape({prompt:'提示词',state:'会话状态',tool_boundary:'工具调用检查',workflow:'执行流程',joint:'代码与提示词'}[r.edit_surface]||r.edit_surface)+' · 门禁 '+(r.val_score==null?'未完成':(r.val_score*100).toFixed(0)+'%')+'</p><details><summary>改进依据与记录</summary><p>'+escape(r.hypothesis)+'</p><code>'+escape(r.commit||'未提交')+'</code></details></div>').join('');
  $('ace-summary').innerHTML='<span class="pill">'+escape(statuses[p.learning.ace.status]||p.learning.ace.status)+'</span><p>当前技能 '+p.skill_count+' 条 · 学习模型 '+escape(p.models.skills)+'</p>';
  $('skill-preview').innerHTML=p.skill_examples.map(x=>'<details><summary>'+escape(x.issue||'业务经验')+'</summary><p>'+escape(x.insight)+'</p></details>').join('');
  renderVersions();
 }catch(e){$('comparison').textContent='实验状态暂时不可用：'+e.message;}
}
$('version-scope').onchange=renderVersions;
init();
