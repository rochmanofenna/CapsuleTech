const API_BASE = window.API_BASE || window.location.origin;

const runListEl = document.getElementById("runList");
const eventsEl = document.getElementById("events");
const runTitleEl = document.getElementById("runTitle");
const runMetaEl = document.getElementById("runMeta");
const toastEl = document.getElementById("toast");
const refreshBtn = document.getElementById("refreshRuns");

let runs = [];
let activeRun = null;

const formatTime = (ts) => new Date(ts).toLocaleString();

async function api(path) {
  const url = `${API_BASE}${path}`;
  const response = await fetch(url, { headers: { accept: "application/json" } });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed (${response.status})`);
  }
  return response.json();
}

function showToast(message) {
  toastEl.textContent = message;
  toastEl.classList.remove("hidden");
  setTimeout(() => toastEl.classList.add("hidden"), 2500);
}

function renderRuns() {
  runListEl.innerHTML = "";
  runs.forEach((run) => {
    const li = document.createElement("li");
    li.className = `run-item${run.run_id === activeRun ? " active" : ""}`;
    li.innerHTML = `
      <strong>${run.run_id}</strong>
      <div><small>${run.backend}</small></div>
      <div><small>${run.policy_id}</small></div>
      <div><small>${formatTime(run.created_at)}</small></div>
    `;
    li.onclick = () => selectRun(run.run_id);
    runListEl.appendChild(li);
  });
}

function renderEvents(events) {
  if (!events || !events.length) {
    eventsEl.innerHTML = "<p>No events yet.</p>";
    return;
  }
  eventsEl.innerHTML = "";
  events.forEach((evt) => {
    const card = document.createElement("div");
    card.className = "event-card";
    card.innerHTML = `
      <div><span class="type">${evt.type}</span> · seq ${evt.seq}</div>
      <div><small>${formatTime(evt.ts_ms)}</small></div>
      <pre>${JSON.stringify(evt.data, null, 2)}</pre>
    `;
    eventsEl.appendChild(card);
  });
}

async function selectRun(runId) {
  activeRun = runId;
  renderRuns();
  runTitleEl.textContent = runId;
  const run = runs.find((r) => r.run_id === runId);
  if (run) {
    runMetaEl.textContent = `${run.policy_id} · ${run.track_id}`;
  } else {
    runMetaEl.textContent = "";
  }
  eventsEl.innerHTML = "<p>Loading events…</p>";
  try {
    const data = await api(`/api/runs/${runId}/events`);
    renderEvents(data.events);
  } catch (err) {
    eventsEl.innerHTML = `<p class="error">${err.message}</p>`;
  }
}

async function loadRuns() {
  runListEl.innerHTML = "<li>Loading…</li>";
  try {
    const data = await api("/api/runs");
    runs = data.runs;
    renderRuns();
    if (runs.length && !activeRun) {
      selectRun(runs[0].run_id);
    }
    showToast("Runs refreshed");
  } catch (err) {
    runListEl.innerHTML = `<li class="error">${err.message}</li>`;
    showToast(err.message);
  }
}

refreshBtn.onclick = loadRuns;
loadRuns();
