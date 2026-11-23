const svc={ticket:null,detail:null,offset:0,article:null,recordTotal:0};
const ticketStatus={open:'处理中',waiting_confirmation:'待确认',human:'人工接管',resolved:'已解决',closed:'已关闭'};
const orderStatus={pending:'待处理',processed:'已处理',delivered:'已送达',cancelled:'已取消','return requested':'退货已申请','exchange requested':'换货已申请'};
async function serviceCall(path,options={}){return api('/service'+path,options);}
async function recordsPage(){
 try{
 const kind=$('record-kind').value;
 const p=new URLSearchParams({kind,q:$('record-query').value,status:kind==='orders'?$('record-status').value:'',offset:svc.offset});
 const r=await serviceCall('/records?'+p);svc.recordTotal=r.total;
 $('record-count').textContent=`共 ${r.total} 条 · 第 ${Math.floor(svc.offset/20)+1} 页`;
 $('record-prev').disabled=svc.offset===0;$('record-next').disabled=svc.offset+20>=r.total;
 $('record-list').innerHTML=r.rows.map(o=>{const d=o.data;
 if(kind==='orders')return '<div class="record-row"><div><b>'+escape(o.id)+'</b> <span class="pill">'+escape(orderStatus[d.status]||d.status)+'</span><p>'+escape(d.user_id)+' · '+escape(d.items.map(i=>i.name).join('、'))+'</p></div><button data-new-ticket="'+escape(o.id)+'">创建售后工单</button></div>';
 if(kind==='users')return '<div class="record-row"><div><b>'+escape(d.name.first_name+' '+d.name.last_name)+'</b><p>'+escape(o.id)+' · '+escape(d.email)+'</p></div><button data-customer-orders="'+escape(o.id)+'">查看订单</button></div>';
 return '<details class="record-row"><summary>'+escape(d.name)+' · '+escape(o.id)+'</summary><pre>'+escape(JSON.stringify(d.variants,null,2))+'</pre></details>';
 }).join('')||'<p class="muted">没有匹配的数据。</p>';
 document.querySelectorAll('[data-new-ticket]').forEach(b=>b.onclick=()=>createServiceTicket(b.dataset.newTicket,b));
 document.querySelectorAll('[data-customer-orders]').forEach(b=>b.onclick=()=>{$('record-kind').value='orders';$('record-query').value=b.dataset.customerOrders;svc.offset=0;recordsPage();});
 }catch(e){$('record-list').textContent=e.message;}
}
$('record-search').onsubmit=e=>{e.preventDefault();svc.offset=0;recordsPage();};
$('record-prev').onclick=()=>{svc.offset=Math.max(0,svc.offset-20);recordsPage();};
$('record-next').onclick=()=>{svc.offset+=20;recordsPage();};
async function createServiceTicket(order,button){
 button.disabled=true;
 try{const t=await serviceCall('/tickets',{method:'POST',body:JSON.stringify({order_id:order,title:'订单 '+order+' 售后咨询',version_id:$('service-version').value||null,mode:$('service-mode').value})});svc.ticket=t.id;sessionStorage.setItem('service-ticket',t.id);navigate('tickets');await serviceRefresh();}
 catch(e){toast(e.message);}finally{button.disabled=false;}
}
async function serviceRefresh(){
 try{
 const [overview,tickets]=await Promise.all([serviceCall('/overview'),serviceCall('/tickets?status='+$('ticket-filter').value)]);
 $('service-stats').innerHTML=stat('待处理工单',(overview.tickets.open||0)+(overview.tickets.waiting_confirmation||0),'Agent 与操作员协作')+stat('待确认方案',overview.pending_approvals,'确认前不修改订单')+stat('人工接管',overview.tickets.human||0,'复杂业务保留上下文')+stat('已解决',(overview.tickets.resolved||0)+(overview.tickets.closed||0),'有操作记录可追溯');
 $('ticket-list').innerHTML=tickets.map(t=>'<button class="ticket-item '+(svc.ticket===t.id?'active':'')+'" data-ticket="'+t.id+'"><b>'+escape(t.title)+'</b><span>'+escape(ticketStatus[t.status])+' · '+escape(t.owner)+' · '+(t.priority==='urgent'?'紧急':'普通')+'</span></button>').join('')||'<p class="muted">暂无工单。从订单创建一张开始。</p>';
 document.querySelectorAll('[data-ticket]').forEach(b=>b.onclick=()=>{svc.ticket=b.dataset.ticket;svc.detail=null;sessionStorage.setItem('service-ticket',svc.ticket);serviceRefresh();});
 if(svc.ticket)await renderTicket();
 }catch(e){toast(e.message);}
}
async function renderTicket(){
 const t=await serviceCall('/tickets/'+svc.ticket);
 const signature=JSON.stringify(t);if(svc.detail===signature)return;
 const oldText=$('service-message')?.value||'', oldNote=$('ticket-note')?.value||'';
 svc.detail=signature;svc.current=t;
 const proposalCards=t.proposals.map(p=>'<div class="proposal"><b>'+escape(p.summary)+'</b><p>操作：'+escape(({cancel:'取消订单',return:'申请退货',exchange:'申请换货',address:'修改地址'}[p.action]||p.action))+' · '+escape({pending:'等待确认',approved:'已执行',rejected:'已驳回',superseded:'已失效'}[p.status])+'</p><pre>'+escape(JSON.stringify(p.parameters,null,2))+'</pre>'+(p.status==='pending'&&!t.routed?'<div class="row"><button class="primary" data-decision="'+p.id+'" data-approve="true" '+(t.active_turn?'disabled':'')+'>确认并执行</button><button data-decision="'+p.id+'" data-approve="false" '+(t.active_turn?'disabled':'')+'>驳回方案</button></div>':'')+'</div>').join('');
 const last=t.turns[0];
 $('ticket-detail').innerHTML='<article class="card"><div class="row"><h2>'+escape(t.title)+'</h2><span class="pill">'+escape(ticketStatus[t.status])+'</span></div><p class="muted">'+escape(t.order_id)+' · '+escape(orderStatus[t.order.status]||t.order.status)+' · '+escape(t.customer.name.first_name+' '+t.customer.name.last_name)+' · 版本 '+escape(t.version_id.slice(0,8))+'</p>'+proposalCards+
 '<div class="messages service-messages">'+t.messages.filter(m=>['assistant','user'].includes(m.role)&&m.content).map(m=>'<div class="bubble '+m.role+'">'+escape(m.content)+'</div>').join('')+(t.messages.length?'':'<p class="muted">告诉 Agent 客户的诉求，例如“查一下订单能否取消”，或“我想退回第一件商品”。Agent 会查询依据，再提出处理方案。</p>')+'</div>'+
 (last?.error?'<p class="note">本轮调用未完成：'+escape(last.error)+'。已有方案仍需核对后确认，可重新发送或转人工。</p>':'')+
 (t.routed?'<p class="small">这张工单来自统一售后客服。请回原会话发送消息和确认方案。</p>':'<form id="service-chat-form" class="chat-input"><textarea id="service-message" rows="3" required placeholder="描述客户诉求或继续追问" '+(t.active_turn?'disabled':'')+'></textarea><button class="primary" '+(t.active_turn||!['open','waiting_confirmation'].includes(t.status)?'disabled':'')+'>'+(t.active_turn?'处理中':'发送给 Agent')+'</button></form>')+(t.active_turn?'<p class="small">'+escape(last?.stage||'等待执行')+'</p>':'')+'</article>'+
 '<article class="card"><h2>人工处理与工单流转</h2><form id="ticket-update" class="fields"><div class="row"><label>负责人<input name="owner" value="'+escape(t.owner)+'" maxlength="80"></label><label>优先级<select name="priority"><option value="normal" '+(t.priority==='normal'?'selected':'')+'>普通</option><option value="urgent" '+(t.priority==='urgent'?'selected':'')+'>紧急</option></select></label></div><label>工单状态<select name="status"><option value="">保持当前状态</option><option value="human">转人工</option><option value="open">重新打开 / 交回 Agent</option><option value="resolved">标记已解决</option><option value="closed">关闭工单</option></select></label><textarea id="ticket-note" name="note" placeholder="处理备注 / 人工交接说明" rows="2"></textarea><button '+(t.active_turn?'disabled':'')+'>保存处理记录</button></form></article>'+
 '<article class="card"><h2>工单记忆与经验</h2>'+(t.memory?'<p class="small">近期保留 '+t.memory.recent_rounds+' 轮；归纳 '+t.memory.covered_messages+' 条历史消息。输入估算 '+t.memory.context_tokens_estimate+' Token（历史上下文），原始历史估算 '+t.memory.full_history_tokens_estimate+' Token（非计费值）。</p><details><summary>客户诉求、约束与进度</summary><pre>'+escape(JSON.stringify({摘要:t.memory.summary,业务状态:t.memory.state,历史客户原文:t.memory.customer_requests},null,2))+'</pre></details><details><summary>本轮选用 '+(t.memory.selected_skills||[]).length+' 条经验</summary><pre>'+escape(JSON.stringify(t.memory.selected_skills,null,2))+'</pre></details>':'<p class="muted">客服完成回复后显示本轮记忆与经验记录。</p>')+'</article>'+
 '<article class="card"><h2>多 Agent 协作记录</h2><p class="muted">主 Agent 按需协调订单核验、政策分析、方案生成和风险审核。方案经过确认后才能执行。</p>'+(t.specialists||[]).map(r=>'<details><summary>'+escape({order_specialist:'订单核验',policy_specialist:'政策分析',solution_specialist:'方案生成',risk_reviewer:'风险审核'}[r.role]||r.role)+' · '+escape(statuses[r.status]||r.status)+' · '+(r.elapsed_ms/1000).toFixed(1)+' 秒</summary><p class="small">'+escape(r.model)+' · '+escape(r.id)+'</p><pre>'+escape(JSON.stringify(r.output,null,2))+'</pre><details><summary>核对输入与引用证据</summary><pre>'+escape(JSON.stringify(r.evidence,null,2))+'</pre></details></details>').join('')+'</article>'+
 '<article class="card"><h2>工具执行与业务审计</h2>'+t.turns.map(r=>'<details><summary>Agent 调用 · '+escape(statuses[r.status]||r.status)+' · '+r.trace.length+' 个工具</summary>'+r.trace.map(x=>'<div class="trace-turn"><b>'+escape(x.tool)+'</b><pre>'+escape(JSON.stringify({参数:x.arguments,结果:x.output},null,2))+'</pre></div>').join('')+'</details>').join('')+t.audit.map(a=>'<details><summary>'+escape(a.at)+' · '+escape(a.action)+' · '+escape(a.actor)+'</summary><pre>'+escape(JSON.stringify(a.details,null,2))+'</pre></details>').join('')+'</article>';
 if($('service-message'))$('service-message').value=oldText;$('ticket-note').value=oldNote;
 const box=document.querySelector('.service-messages');box.scrollTop=box.scrollHeight;
 if($('service-chat-form'))$('service-chat-form').onsubmit=async e=>{e.preventDefault();const b=e.target.querySelector('button');b.disabled=true;try{await serviceCall('/tickets/'+t.id+'/messages',{method:'POST',headers:{'Idempotency-Key':crypto.randomUUID()},body:JSON.stringify({message:$('service-message').value})});$('service-message').value='';await serviceRefresh();}catch(err){toast(err.message);b.disabled=false;}};
 $('ticket-update').onsubmit=async e=>{e.preventDefault();const values=Object.fromEntries(new FormData(e.target));if(!values.status)delete values.status;try{await serviceCall('/tickets/'+t.id,{method:'PATCH',body:JSON.stringify({...values,revision:t.revision})});await serviceRefresh();}catch(err){toast(err.message);}};
 document.querySelectorAll('[data-decision]').forEach(b=>b.onclick=async()=>{b.disabled=true;try{await serviceCall('/proposals/'+b.dataset.decision+'/decision',{method:'POST',body:JSON.stringify({approved:b.dataset.approve==='true',revision:t.revision})});await serviceRefresh();}catch(e){toast(e.message);b.disabled=false;}});
}
$('ticket-filter').onchange=serviceRefresh;
async function knowledgePage(){
 try{svc.articles=await serviceCall('/knowledge');$('knowledge-list').innerHTML=svc.articles.map(d=>'<button class="ticket-item" data-doc="'+d.id+'"><b>'+escape(d.title)+'</b><span>'+escape({draft:'草稿',published:'已发布',archived:'已归档'}[d.status])+' · 修订 '+d.revision+' · '+escape(d.tags)+'</span></button>').join('');document.querySelectorAll('[data-doc]').forEach(b=>b.onclick=()=>editKnowledge(svc.articles.find(d=>d.id===b.dataset.doc)));}catch(e){toast(e.message);}
}
function editKnowledge(d){svc.article=d;const f=$('knowledge-form');for(const key of ['title','content','tags','status'])f.elements[key].value=d?d[key]:(key==='status'?'draft':'');f.elements.order_states.value=(d?.order_states||[]).join(',');f.elements.product_ids.value=(d?.product_ids||[]).join(',');$('knowledge-edit-state').textContent=d?'编辑 '+d.id+' · 修订 '+d.revision:'新建文档';}
$('new-article').onclick=()=>editKnowledge(null);
$('knowledge-form').onsubmit=async e=>{e.preventDefault();const b=e.target.querySelector('button');b.disabled=true;try{const values={...Object.fromEntries(new FormData(e.target)),revision:svc.article?.revision||0};values.order_states=values.order_states.split(',').map(x=>x.trim()).filter(Boolean);values.product_ids=values.product_ids.split(',').map(x=>x.trim()).filter(Boolean);const result=await serviceCall('/knowledge'+(svc.article?'/'+svc.article.id:''),{method:svc.article?'PUT':'POST',body:JSON.stringify(values)});await knowledgePage();editKnowledge(svc.articles.find(a=>a.id===result.id));toast('知识已保存');}catch(err){toast(err.message);}finally{b.disabled=false;}};
svc.ticket=sessionStorage.getItem('service-ticket');
recordsPage();serviceRefresh();knowledgePage();
setInterval(()=>{if(!$('tickets').classList.contains('hidden'))serviceRefresh();},5000);

$('knowledge-search-form').onsubmit=async e=>{e.preventDefault();const button=e.target.querySelector('button');button.disabled=true;$('knowledge-results').textContent='正在查找适用政策…';try{const r=await serviceCall('/knowledge/search',{method:'POST',body:JSON.stringify({query:$('knowledge-query').value,order_id:$('knowledge-order').value||null})});$('knowledge-results').innerHTML=r.sources.map(d=>'<details open><summary>'+escape(d.title)+' · '+escape(d.id)+'</summary><p>'+escape(d.content)+'</p><p class="small">相关性 '+d.retrieval.relevance+'/3 · 语义相似度 '+d.retrieval.cosine.toFixed(3)+'</p></details>').join('')||'没有足够相关的已发布依据，请补充问题或订单。';}catch(err){$('knowledge-results').textContent=err.message;}finally{button.disabled=false;}};
