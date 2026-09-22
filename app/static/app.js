const state = {
  sessionId: null,
  dataset: null,
  running: false,
  controller: null,
  turn: null,
  usage: null,
  replacing: false,
};

const $ = (selector) => document.querySelector(selector);
const conversation = $("#conversation");
const messages = $("#messages");
const input = $("#message-input");
const sendButton = $("#send-button");
const cancelButton = $("#cancel-button");

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 3200);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    const retryAfter = Number(response.headers.get("Retry-After") || 0);
    const error = new Error(retryAfter ? `${detail} Try again in about ${retryAfter} seconds.` : detail);
    error.status = response.status;
    error.retryAfter = retryAfter;
    throw error;
  }
  return response;
}

function renderUsage(usage) {
  if (!usage) return;
  state.usage = usage;
  const status = $("#usage-status");
  status.textContent = `Analysis requests remaining: ${usage.remaining} of ${usage.limit}`;
}

async function replaceSession(previousSessionId = null) {
  const response = await api("/api/sessions/replace", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ previous_session_id: previousSessionId || null }),
  });
  const payload = await response.json();
  state.sessionId = payload.session_id;
  window.sessionStorage.setItem("dataScienceAgentSessionId", state.sessionId);
  renderUsage(payload.usage);
}

async function loadSamples() {
  const { samples } = await (await api("/api/samples")).json();
  const list = $("#sample-list");
  list.replaceChildren();
  samples.forEach((sample) => {
    const button = document.createElement("button");
    button.className = "sample-button";
    const name = document.createElement("span");
    name.textContent = sample.name;
    const type = document.createElement("small");
    type.textContent = sample.task_type;
    button.append(name, type);
    button.addEventListener("click", () => chooseSample(sample.id));
    list.append(button);
  });
}

function confirmReplacement() {
  return !state.dataset || window.confirm("Replace the active dataset? Current analysis results will be cleared.");
}

async function chooseSample(sampleId) {
  if (!confirmReplacement()) return;
  const replacing = Boolean(state.dataset);
  try {
    const response = await api(`/api/sessions/${state.sessionId}/datasets/sample`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_id: sampleId, confirm_replace: Boolean(state.dataset) }),
    });
    if (replacing) clearConversation();
    setDataset(await response.json());
    showToast("Sample dataset loaded");
  } catch (error) { showToast(error.message); }
}

async function uploadCsv(file) {
  if (!file || !confirmReplacement()) return;
  const replacing = Boolean(state.dataset);
  const form = new FormData();
  form.append("file", file);
  try {
    const confirmed = state.dataset ? "?confirm_replace=true" : "";
    const response = await api(`/api/sessions/${state.sessionId}/datasets/upload${confirmed}`, {
      method: "POST", body: form,
    });
    if (replacing) clearConversation();
    setDataset(await response.json());
    showToast("CSV loaded successfully");
  } catch (error) { showToast(error.message); }
  finally { $("#csv-input").value = ""; }
}

function setDataset(dataset) {
  state.dataset = dataset;
  $("#dataset-card").classList.remove("is-hidden");
  $("#dataset-status").classList.add("active");
  $("#dataset-name").textContent = dataset.name;
  $("#row-count").textContent = Number(dataset.rows).toLocaleString();
  $("#column-count").textContent = Number(dataset.columns).toLocaleString();
  const chips = $("#column-chips");
  chips.replaceChildren();
  dataset.schema.slice(0, 18).forEach((column) => {
    const chip = document.createElement("span");
    chip.className = "column-chip";
    chip.textContent = column.column || column.name;
    chip.title = `${column.dtype}${column.missing ? ` · ${column.missing} missing` : ""}`;
    chips.append(chip);
  });
  input.disabled = false;
  input.placeholder = "Ask a question about your data…";
  sendButton.disabled = !input.value.trim();
  document.querySelectorAll(".prompt-card").forEach((button) => button.disabled = false);
}

function clearConversation() {
  messages.replaceChildren();
  state.turn = null;
  $("#welcome").classList.remove("is-hidden");
}

function scrollToBottom() {
  conversation.scrollTo({ top: conversation.scrollHeight, behavior: "smooth" });
}

function markdownText(text) {
  return text
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/__(.*?)__/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*(.*?)\*/g, "$1");
}

function tableCells(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => markdownText(cell.trim()));
}

function renderMarkdown(container, text) {
  const lines = text.split("\n");
  let index = 0;
  while (index < lines.length) {
    const line = lines[index].trim();
    if (!line) { index += 1; continue; }
    if (/^---+$/.test(line)) {
      container.append(document.createElement("hr")); index += 1; continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const element = document.createElement(`h${Math.min(heading[1].length + 1, 4)}`);
      element.textContent = markdownText(heading[2]); container.append(element); index += 1; continue;
    }
    if (line.includes("|") && index + 1 < lines.length && /^\s*\|?\s*:?-+/.test(lines[index + 1])) {
      const headers = tableCells(line); index += 2;
      const wrap = document.createElement("div"); wrap.className = "table-scroll";
      const table = document.createElement("table");
      const thead = document.createElement("thead"); const tr = document.createElement("tr");
      headers.forEach((value) => { const th = document.createElement("th"); th.textContent = value; tr.append(th); });
      thead.append(tr); table.append(thead); const tbody = document.createElement("tbody");
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        const row = document.createElement("tr");
        tableCells(lines[index]).forEach((value) => { const td = document.createElement("td"); td.textContent = value; row.append(td); });
        tbody.append(row); index += 1;
      }
      table.append(tbody); wrap.append(table); container.append(wrap); continue;
    }
    const bullet = line.match(/^[-*]\s+(.+)$/);
    const numbered = line.match(/^\d+[.)]\s+(.+)$/);
    if (bullet || numbered) {
      const list = document.createElement(bullet ? "ul" : "ol");
      while (index < lines.length) {
        const match = lines[index].trim().match(bullet ? /^[-*]\s+(.+)$/ : /^\d+[.)]\s+(.+)$/);
        if (!match) break;
        const item = document.createElement("li"); item.textContent = markdownText(match[1]); list.append(item); index += 1;
      }
      container.append(list); continue;
    }
    const paragraphLines = [line]; index += 1;
    while (index < lines.length && lines[index].trim() && !/^(#{1,4})\s+|^[-*]\s+|^\d+[.)]\s+|^---+$/.test(lines[index].trim())) {
      if (lines[index].includes("|") && index + 1 < lines.length && /^\s*\|?\s*:?-+/.test(lines[index + 1])) break;
      paragraphLines.push(lines[index].trim()); index += 1;
    }
    const paragraph = document.createElement("p");
    paragraph.textContent = markdownText(paragraphLines.join(" ")); container.append(paragraph);
  }
}

function addMessage(role, text) {
  $("#welcome").classList.add("is-hidden");
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "Y" : "AI";
  const body = document.createElement("div");
  body.className = "message-body";
  if (role === "assistant") renderMarkdown(body, text);
  else body.textContent = text;
  wrapper.append(avatar, body);
  messages.append(wrapper);
  scrollToBottom();
  return { wrapper, body };
}

function addCard(title, className = "", parent = messages) {
  $("#welcome").classList.add("is-hidden");
  const card = document.createElement("section");
  card.className = `event-card ${className}`;
  if (title) {
    const heading = document.createElement("h3");
    heading.textContent = title;
    card.append(heading);
  }
  parent.append(card);
  return card;
}

function createActivity() {
  $("#welcome").classList.add("is-hidden");
  const card = document.createElement("details");
  card.className = "activity-card";
  card.open = true;
  const summary = document.createElement("summary");
  summary.textContent = "Preparing analysis…";
  const list = document.createElement("ul");
  list.className = "activity-list";
  card.append(summary, list);
  messages.append(card);
  state.turn = { card, summary, list, entries: [], artifacts: [], toolCount: 0 };
  addActivity("Understanding your request", false);
}

function addActivity(label, isTool = false) {
  const turn = state.turn;
  if (!turn || !label) return;
  const normalized = String(label).replace(/…+$/, "").trim();
  if (turn.entries.at(-1)?.label === normalized) return;
  const active = turn.list.querySelector("li.active");
  if (active) active.className = "complete";
  const item = document.createElement("li");
  item.className = "active";
  item.textContent = normalized;
  turn.list.append(item);
  turn.entries.push({ label: normalized, isTool });
  if (isTool) turn.toolCount += 1;
  turn.summary.textContent = normalized;
  scrollToBottom();
}

function finishActivity(outcome = "complete") {
  const turn = state.turn;
  if (!turn) return;
  const active = turn.list.querySelector("li.active");
  if (active) active.className = outcome === "complete" ? "complete" : "pending";
  turn.card.classList.add(outcome);
  const count = turn.toolCount;
  if (outcome === "cancelled") turn.summary.textContent = "Analysis cancelled";
  else if (outcome === "failed") turn.summary.textContent = "Analysis could not be completed";
  else turn.summary.textContent = `Analysis completed · ${count} tool${count === 1 ? "" : "s"} used`;
  turn.card.open = outcome !== "complete";
}

function renderTable(card, payload) {
  let rows = payload?.rows;
  if (!Array.isArray(rows) && payload?.data?.rows) rows = payload.data.rows;
  if (!Array.isArray(rows) && payload?.pairs) rows = payload.pairs;
  if (!Array.isArray(rows)) {
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(payload, null, 2);
    pre.style.whiteSpace = "pre-wrap";
    card.append(pre);
    return;
  }
  if (!rows.length) {
    const empty = document.createElement("p");
    empty.textContent = "No rows to display.";
    card.append(empty);
    return;
  }
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const wrap = document.createElement("div");
  wrap.className = "table-scroll";
  const table = document.createElement("table");
  const head = document.createElement("thead");
  const headerRow = document.createElement("tr");
  columns.forEach((column) => {
    const th = document.createElement("th"); th.textContent = column; headerRow.append(th);
  });
  head.append(headerRow); table.append(head);
  const body = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((column) => {
      const td = document.createElement("td");
      const value = row[column];
      td.textContent = value == null ? "—" : typeof value === "object" ? JSON.stringify(value) : String(value);
      td.title = td.textContent;
      tr.append(td);
    });
    body.append(tr);
  });
  table.append(body); wrap.append(table); card.append(wrap);
}

function renderModel(event, parent = messages) {
  const card = addCard(`Baseline model · ${event.target}`, "final-artifact model-artifact", parent);
  const intro = document.createElement("p");
  intro.textContent = `${event.problem_type} · selected ${event.selected_model.replaceAll("_", " ")}`;
  card.append(intro);
  const grid = document.createElement("div"); grid.className = "metric-grid";
  Object.entries(event.models[event.selected_model] || {}).forEach(([label, value]) => {
    const metric = document.createElement("div"); metric.className = "metric";
    const name = document.createElement("span"); name.textContent = label.replaceAll("_", " ");
    const number = document.createElement("strong"); number.textContent = value;
    metric.append(name, number); grid.append(metric);
  });
  card.append(grid);
  if (event.url || event?.chart?.url) {
    const image = document.createElement("img"); image.className = "chart-image";
    image.src = event.url || event.chart.url; image.alt = "Model evaluation chart"; card.append(image);
  }
}

function renderProposal(event, parent = messages) {
  const card = addCard("Review cleaning proposal", "final-artifact proposal-artifact", parent);
  const summary = document.createElement("p");
  summary.textContent = `${event.operation.replaceAll("_", " ")} · ${event.before_rows} → ${event.after_rows} rows`;
  card.append(summary);
  if (event.warnings?.length) {
    const warning = document.createElement("p"); warning.textContent = event.warnings.join(" "); card.append(warning);
  }
  const actions = document.createElement("div"); actions.className = "proposal-actions";
  const apply = document.createElement("button"); apply.className = "primary-small"; apply.textContent = "Apply change";
  const reject = document.createElement("button"); reject.className = "secondary-small"; reject.textContent = "Discard";
  apply.disabled = state.running;
  reject.disabled = state.running;
  apply.addEventListener("click", async () => {
    try {
      const response = await api(`/api/sessions/${state.sessionId}/transformations/${event.proposal_id}/apply`, { method: "POST" });
      const result = await response.json(); setDataset(result.dataset); actions.replaceChildren("Applied");
    } catch (error) { showToast(error.message); }
  });
  reject.addEventListener("click", async () => {
    try {
      await api(`/api/sessions/${state.sessionId}/transformations/${event.proposal_id}`, { method: "DELETE" });
      actions.replaceChildren("Discarded");
    } catch (error) { showToast(error.message); }
  });
  actions.append(apply, reject); card.append(actions);
}

function renderFinalResponse(message, artifacts) {
  const { body } = addMessage("assistant", message);
  artifacts.forEach((event) => {
    if (event.event === "chart") {
      const card = addCard(event.title || "Chart", "final-artifact chart-artifact", body);
      const image = document.createElement("img"); image.className = "chart-image";
      image.src = event.url; image.alt = event.title || "Analysis chart"; card.append(image);
    } else if (event.event === "model_result") {
      renderModel(event, body);
    } else if (event.event === "transformation_proposal") {
      renderProposal(event, body);
    }
  });
}

function renderEvent(event) {
  if (event.event === "status") {
    addActivity(event.message, false);
  } else if (event.event === "tool_status") {
    addActivity(event.message, true);
  } else if (event.event === "analysis_plan") {
    addActivity("Planning the analysis", true);
  } else if (["table", "chart", "model_result", "transformation_proposal", "warning"].includes(event.event)) {
    state.turn?.artifacts.push(event);
  } else if (event.event === "message") {
    const artifacts = state.turn?.artifacts || [];
    finishActivity("complete");
    renderFinalResponse(event.message, artifacts);
    state.turn = null;
  } else if (event.event === "error") {
    finishActivity("failed");
    const card = addCard("Could not complete analysis", "error");
    const text = document.createElement("p"); text.textContent = event.message; card.append(text);
    state.turn = null;
  } else if (event.event === "turn_complete") {
    renderUsage(event.usage);
  }
  scrollToBottom();
}

function setRunning(running) {
  state.running = running;
  input.disabled = running || !state.dataset;
  sendButton.disabled = running || !input.value.trim() || !state.dataset;
  cancelButton.classList.toggle("is-hidden", !running);
  document.querySelectorAll(".proposal-actions button").forEach((button) => { button.disabled = running; });
}

async function sendMessage(message) {
  if (!message.trim() || state.running || !state.dataset) return;
  addMessage("user", message.trim());
  createActivity();
  input.value = ""; input.style.height = "auto"; setRunning(true);
  state.controller = new AbortController();
  try {
    const response = await api(`/api/sessions/${state.sessionId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message.trim() }),
      signal: state.controller.signal,
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n"); buffer = lines.pop() || "";
      lines.filter(Boolean).forEach((line) => renderEvent(JSON.parse(line)));
      if (done) break;
    }
    if (buffer.trim()) renderEvent(JSON.parse(buffer));
  } catch (error) {
    if (error.name !== "AbortError") renderEvent({ event: "error", message: error.message });
  } finally {
    setRunning(false); state.controller = null; input.focus();
  }
}

async function resetDataset() {
  if (!state.dataset || !window.confirm("Restore the original dataset and discard cleaning changes?")) return;
  try {
    const response = await api(`/api/sessions/${state.sessionId}/dataset/reset`, { method: "POST" });
    setDataset(await response.json()); showToast("Original dataset restored");
  } catch (error) { showToast(error.message); }
}

async function newSession() {
  if (state.replacing) return;
  const warning = state.running
    ? "An analysis is still running. Cancel it and start a new temporary session?"
    : "Start a new temporary session?";
  if ((state.running || state.dataset) && !window.confirm(warning)) return;
  state.replacing = true;
  $("#clear-button").disabled = true;
  const previousSessionId = state.sessionId;
  try {
    if (state.running) {
      try { await api(`/api/sessions/${previousSessionId}/chat/cancel`, { method: "POST" }); } catch (_) {}
      state.controller?.abort();
      finishActivity("cancelled");
      state.turn = null;
    }
    await replaceSession(previousSessionId);
    state.dataset = null; messages.replaceChildren();
    $("#welcome").classList.remove("is-hidden"); $("#dataset-card").classList.add("is-hidden");
    $("#dataset-status").classList.remove("active"); input.disabled = true; sendButton.disabled = true;
    input.placeholder = "Load a dataset to begin…";
    document.querySelectorAll(".prompt-card").forEach((button) => button.disabled = true);
    showToast("New session ready");
  } catch (error) {
    showToast(error.message);
  } finally {
    state.replacing = false;
    $("#clear-button").disabled = false;
  }
}

$("#csv-input").addEventListener("change", (event) => uploadCsv(event.target.files[0]));
$("#reset-button").addEventListener("click", resetDataset);
$("#clear-button").addEventListener("click", newSession);
$("#composer").addEventListener("submit", (event) => { event.preventDefault(); sendMessage(input.value); });
input.addEventListener("input", () => {
  input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight, 130)}px`;
  sendButton.disabled = state.running || !input.value.trim() || !state.dataset;
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendMessage(input.value); }
});
cancelButton.addEventListener("click", async () => {
  try { await api(`/api/sessions/${state.sessionId}/chat/cancel`, { method: "POST" }); } catch (_) {}
  state.controller?.abort(); finishActivity("cancelled"); state.turn = null;
});
document.querySelectorAll(".prompt-card").forEach((button) => {
  button.disabled = true;
  button.addEventListener("click", () => sendMessage(button.dataset.prompt));
});

const previousSessionId = window.sessionStorage.getItem("dataScienceAgentSessionId");
Promise.all([replaceSession(previousSessionId), loadSamples()]).catch((error) => showToast(error.message));
