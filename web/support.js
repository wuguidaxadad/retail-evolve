let supportSession = JSON.parse(sessionStorage.getItem("retail-support") || "null");
let supportSnapshot = "";
let supportLoading = false;
let pendingSupportMessage;
let clarificationTarget;

function supportPath(path = "") {
  return "/support/conversations/" + encodeURIComponent(supportSession.id) + path;
}

function supportRequest(path, options = {}) {
  return api(supportPath(path), {...options, headers: {"Support-Token": supportSession.token}});
}

async function refreshSupport() {
  if (!supportSession || supportLoading) return;
  supportLoading = true;
  const activeSession = supportSession;
  try {
    const data = await supportRequest("");
    if (supportSession !== activeSession) return;
    const snapshot = JSON.stringify(data);
    if (snapshot === supportSnapshot) return;
    supportSnapshot = snapshot;
    $("support-identity").textContent = data.customer_id;
    $("support-send").classList.remove("hidden");
    const stages = {queued:"等待分派",running:"正在分派",failed:"分派失败",clarification:"等待补充"};
    $("support-messages").innerHTML = data.messages.map(message =>
      '<div class="bubble user">' + escape(message.text) + '</div>' +
      (message.reply ? '<div class="bubble assistant">' + escape(message.reply) + '</div>' : '') +
      (stages[message.status] ? '<p class="small">' + stages[message.status] + '</p>' : '') +
      (message.status === "failed" ? '<button data-support-retry="' + escape(message.id) + '">重试分派</button>' : "") +
      (message.status === "clarification" ? '<button data-support-clarify="' + escape(message.id) + '">补充这条消息</button>' : "") +
      (["failed","clarification"].includes(message.status) ? '<button data-support-discard="' +
        escape(message.id) + '">忽略未处理内容</button>' : "")
    ).join("");
    $("support-tasks").innerHTML = data.tasks.map(task =>
      '<article class="card"><h2>' + escape(task.title) + '</h2><p>' +
      escape(task.working ? "正在处理" : (task.error || ticketStatus[task.status] || task.status)) + '</p>' +
      task.replies.map(reply => '<div class="bubble assistant">' + escape(reply) + '</div>').join("") +
      (task.proposal ? '<div class="proposal"><b>' + escape(task.proposal.summary) + '</b><pre>' +
        escape(JSON.stringify(task.proposal.parameters, null, 2)) + '</pre>' +
        '<button class="primary" data-support-proposal="' + escape(task.proposal.id) +
        '" data-revision="' + task.revision + '">确认这份方案</button></div>' : "") + '</article>'
    ).join("");
    document.querySelectorAll("[data-support-retry]").forEach(button => {
      button.onclick = () => submitSupportAction(button, "/messages/" + button.dataset.supportRetry + "/retry");
    });
    document.querySelectorAll("[data-support-discard]").forEach(button => {
      button.onclick = () => submitSupportAction(button, "/messages/" + button.dataset.supportDiscard + "/discard");
    });
    document.querySelectorAll("[data-support-clarify]").forEach(button => {
      button.onclick = () => {
        clarificationTarget = button.dataset.supportClarify;
        $("support-clarifying").classList.remove("hidden");
        $("support-send").elements.message.focus();
      };
    });
    document.querySelectorAll("[data-support-proposal]").forEach(button => {
      button.onclick = () => submitSupportAction(button, "/proposals/" + button.dataset.supportProposal + "/decision",
        {approved:true, revision:Number(button.dataset.revision)});
    });
  } catch (error) {
    $("support-identity").textContent = error.message;
  } finally {
    supportLoading = false;
  }
}

async function submitSupportAction(button, path, body) {
  button.disabled = true;
  try {
    const result = await supportRequest(path, {method:"POST", body:JSON.stringify(body)});
    if (result.status === "superseded") toast("这份方案已失效，请查看当前办理进度。");
    await refreshSupport();
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
  }
}

$("support-start").onsubmit = async event => {
  event.preventDefault();
  const form = event.target;
  const button = form.querySelector("button");
  button.disabled = true;
  try {
    supportSession = await api("/support/conversations", {
      method:"POST", body:JSON.stringify({customer_id:form.elements.customer_id.value.trim()})
    });
    sessionStorage.setItem("retail-support", JSON.stringify(supportSession));
    pendingSupportMessage = undefined;
    $("support-cancel-clarify").click();
    supportSnapshot = "";
    await refreshSupport();
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
  }
};

$("support-send").onsubmit = async event => {
  event.preventDefault();
  const form = event.target;
  const box = form.elements.message;
  const text = box.value;
  const button = form.querySelector("button");
  button.disabled = true;
  if (pendingSupportMessage?.message !== text || pendingSupportMessage?.clarifies_id !== clarificationTarget) {
    pendingSupportMessage = {message:text, clarifies_id:clarificationTarget, key:crypto.randomUUID()};
  }
  try {
    await supportRequest("/messages", {method:"POST", body:JSON.stringify(pendingSupportMessage)});
    pendingSupportMessage = undefined;
    $("support-cancel-clarify").click();
    if (box.value === text) box.value = "";
    await refreshSupport();
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
  }
};

$("support-cancel-clarify").onclick = () => {
  clarificationTarget = undefined;
  $("support-clarifying").classList.add("hidden");
};

setInterval(() => {
  if (!$("support").classList.contains("hidden")) refreshSupport();
}, 1200);
