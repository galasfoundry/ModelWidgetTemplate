// Model picker: lists /api/models, downloads with POST /api/models/<id>/download
// (a background job polled via /api/models/downloads), and switches with
// POST /api/models/select. Relies on loadInfo() from app.js after a switch.
// Deliberately bare: UI polish is pending.
const pickerList = document.querySelector("#picker-list");
const pickerExcluded = document.querySelector("#picker-excluded");
const pickerExcludedWrap = document.querySelector("#picker-excluded-wrap");
const pickerCaps = document.querySelector("#picker-caps");
const pickerStatus = document.querySelector("#picker-status");
let pickerBusy = false;
let pollTimer = null;

function capsLine(caps) {
  const parts = [];
  if (caps.ram_total_gb != null) parts.push(`${caps.ram_total_gb} GB RAM`);
  if (caps.ram_available_gb != null) parts.push(`${caps.ram_available_gb} GB free`);
  if (caps.disk_free_mb != null) parts.push(`${Math.round(caps.disk_free_mb / 1024 * 10) / 10} GB disk free`);
  parts.push(caps.cuda_available ? "CUDA GPU" : "CPU only");
  return parts.join(" · ");
}

function modelItem(model, selectable) {
  const item = document.createElement("li");
  item.className = "picker-item" + (model.active ? " active" : "");
  const head = document.createElement("div");
  head.className = "picker-head";
  const name = document.createElement("strong");
  name.textContent = model.name;
  head.appendChild(name);
  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = (model.active ? "active · " : "") + (model.installed ? "on disk" : `not downloaded (~${model.approx_size_mb} MB)`);
  head.appendChild(badge);
  item.appendChild(head);

  const desc = document.createElement("p");
  desc.className = "extra";
  const origin = model.source === "user" ? " · added from Discover" : "";
  desc.textContent = `${model.task} · ${model.backend}${origin} — ${model.description || ""}`;
  item.appendChild(desc);

  const notes = [...(model.reasons || []), ...(model.warnings || []).map((w) => `warning: ${w}`)];
  if (notes.length) {
    const why = document.createElement("p");
    why.className = "extra";
    why.textContent = notes.join("; ");
    item.appendChild(why);
  }

  const job = model.download;
  const downloading = job && (job.state === "queued" || job.state === "downloading");
  const progress = document.createElement("p");
  progress.className = "extra download-status";
  progress.dataset.modelId = model.id;
  progress.textContent = jobLine(job);
  item.appendChild(progress);

  if (!model.installed && model.downloadable && !downloading) {
    const dl = document.createElement("button");
    dl.type = "button";
    dl.textContent = job && job.state === "error" ? "Retry download" : `Download (~${model.approx_size_mb} MB)`;
    dl.addEventListener("click", () => startDownload(model, dl));
    item.appendChild(dl);
  }
  if (selectable && model.installed && !model.active) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Use this model";
    button.addEventListener("click", () => selectModel(model, button));
    item.appendChild(button);
  }
  if (model.source === "user" && !model.active) {
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Remove from catalog";
    remove.addEventListener("click", () => removeUserModel(model, remove));
    item.appendChild(remove);
  }
  return item;
}

function jobLine(job) {
  if (!job) return "";
  if (job.state === "queued") return "Download queued…";
  if (job.state === "downloading") {
    const total = job.mb_total != null ? ` / ${job.mb_total} MB` : " MB";
    const pct = job.percent != null ? ` (${job.percent}%)` : "";
    return `Downloading ${job.mb_done}${total}${pct}${job.current_file ? " · " + job.current_file : ""}`;
  }
  if (job.state === "error") return `Download failed: ${job.error}`;
  if (job.state === "done") return job.message || "Downloaded.";
  return job.message || "";
}

async function startDownload(model, button) {
  button.disabled = true;
  pickerStatus.textContent = `Starting download of ${model.name}…`;
  try {
    const response = await fetch(`/api/models/${encodeURIComponent(model.id)}/download`, { method: "POST" });
    const payload = await response.json();
    pickerStatus.textContent = response.ok
      ? `${model.name}: ${payload.message || payload.state}`
      : `${model.name}: ${payload.error || "download refused"}`;
  } catch (error) {
    pickerStatus.textContent = "Could not reach the local server.";
  }
  await loadModels();
}

async function pollDownloads() {
  pollTimer = null;
  let payload;
  try {
    const response = await fetch("/api/models/downloads");
    payload = await response.json();
  } catch (error) {
    return;
  }
  let finished = false;
  for (const job of payload.jobs || []) {
    const line = document.querySelector(`.download-status[data-model-id="${CSS.escape(job.id)}"]`);
    if (line) {
      const before = line.dataset.state;
      line.textContent = jobLine(job);
      line.dataset.state = job.state;
      if (before && before !== job.state && (job.state === "done" || job.state === "error")) finished = true;
    }
  }
  if (finished) {
    await loadModels();
  } else if (payload.active) {
    pollTimer = setTimeout(pollDownloads, 1000);
  }
}

async function loadModels() {
  let payload;
  try {
    const response = await fetch("/api/models");
    payload = await response.json();
    if (!response.ok) {
      pickerCaps.textContent = payload.error || "Could not read the model catalog.";
      return;
    }
  } catch (error) {
    pickerCaps.textContent = "Could not reach the local server.";
    return;
  }
  pickerCaps.textContent = capsLine(payload.capabilities);
  pickerList.replaceChildren(...payload.models.map((m) => modelItem(m, !payload.weights_override)));
  pickerExcluded.replaceChildren(...payload.excluded.map((m) => modelItem(m, false)));
  pickerExcludedWrap.hidden = payload.excluded.length === 0;
  for (const line of document.querySelectorAll(".download-status")) {
    const row = [...payload.models, ...payload.excluded].find((m) => m.id === line.dataset.modelId);
    if (row && row.download) line.dataset.state = row.download.state;
  }
  if (payload.weights_override) {
    pickerStatus.textContent = "MODEL_WEIGHTS is set, so switching is turned off (downloads still work).";
  }
  const active = [...payload.models, ...payload.excluded].some(
    (m) => m.download && (m.download.state === "queued" || m.download.state === "downloading"),
  );
  if (active && !pollTimer) pollTimer = setTimeout(pollDownloads, 1000);
}

async function selectModel(model, button) {
  if (pickerBusy) return;
  pickerBusy = true;
  for (const b of document.querySelectorAll("#model-picker button")) b.disabled = true;
  pickerStatus.textContent = `Loading ${model.name}…`;
  try {
    const response = await fetch("/api/models/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: model.id }),
    });
    const payload = await response.json();
    if (!response.ok) {
      pickerStatus.textContent = payload.error || "The switch failed. The previous model is still active.";
    } else {
      pickerStatus.textContent = `Now using ${model.name}.` + (payload.warning ? ` ${payload.warning}` : "");
      showEmpty();
      await loadInfo();
    }
  } catch (error) {
    pickerStatus.textContent = "Could not reach the local server.";
  } finally {
    pickerBusy = false;
    await loadModels();
  }
}

loadModels();


async function removeUserModel(model, button) {
  button.disabled = true;
  pickerStatus.textContent = `Removing ${model.name} from the user catalog…`;
  try {
    const response = await fetch("/api/models/discover/remove", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: model.id }),
    });
    const payload = await response.json();
    pickerStatus.textContent = response.ok
      ? `Removed ${model.name}. The file on disk is unchanged.`
      : (payload.error || "Could not remove that model.");
  } catch (error) {
    pickerStatus.textContent = "Could not reach the local server.";
  }
  await loadModels();
}

const discoverForm = document.querySelector("#discover-form");
const discoverQ = document.querySelector("#discover-q");
const discoverList = document.querySelector("#discover-list");
const discoverStatus = document.querySelector("#discover-status");
const discoverSubmit = document.querySelector("#discover-submit");

function ageLabel(seconds) {
  if (seconds == null) return "";
  if (seconds < 60) return "less than a minute ago";
  if (seconds < 3600) {
    const n = Math.round(seconds / 60);
    return n === 1 ? "1 minute ago" : `${n} minutes ago`;
  }
  const hours = Math.round(seconds / 3600);
  if (hours < 48) return hours === 1 ? "1 hour ago" : `${hours} hours ago`;
  const days = Math.round(hours / 24);
  return days === 1 ? "1 day ago" : `${days} days ago`;
}

function discoverStatusText(payload) {
  if (payload.online) {
    const n = (payload.results || []).length;
    return n ? `${n} model(s) for "${payload.query}".` : `No GGUF chat models found for "${payload.query}".`;
  }
  const age = ageLabel(payload.cache_age_seconds);
  const cacheBit = payload.cached
    ? ` Showing a cached search for "${payload.query}"${age ? " from " + age : ""}.`
    : " No cached search is available.";
  return (payload.error || "Hugging Face could not be reached.") + cacheBit;
}

function discoverItem(row) {
  const item = document.createElement("li");
  item.className = "picker-item";
  const head = document.createElement("div");
  head.className = "picker-head";
  const name = document.createElement("strong");
  name.textContent = row.name || row.hf_id;
  head.appendChild(name);
  const badge = document.createElement("span");
  badge.className = "badge fit-" + (row.fit === "won't fit" ? "wont" : row.fit || "");
  badge.textContent = `${row.fit || "?"} · ${row.quant || "?"} · ~${row.approx_size_mb} MB`;
  head.appendChild(badge);
  item.appendChild(head);

  const desc = document.createElement("p");
  desc.className = "extra";
  const bits = [
    row.hf_id,
    row.license ? `licence ${row.license}` : null,
    row.gated ? "gated" : "public",
    `${row.downloads || 0} downloads`,
    `${row.likes || 0} likes`,
  ].filter(Boolean);
  desc.textContent = bits.join(" · ");
  item.appendChild(desc);

  const note = row.warning || row.reason;
  if (note) {
    const why = document.createElement("p");
    why.className = "extra";
    why.textContent = note;
    item.appendChild(why);
  }

  if (row.in_catalog) {
    const note = document.createElement("p");
    note.className = "extra";
    note.textContent = `Already in the catalog as ${row.in_catalog}.`;
    item.appendChild(note);
  } else if (row.fit !== "won't fit") {
    const add = document.createElement("button");
    add.type = "button";
    add.textContent = "Add to catalog";
    add.addEventListener("click", () => addDiscovered(row, add));
    item.appendChild(add);
  }
  return item;
}

async function addDiscovered(row, button) {
  button.disabled = true;
  discoverStatus.textContent = `Adding ${row.name}…`;
  try {
    const response = await fetch("/api/models/discover/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(row),
    });
    const payload = await response.json();
    if (!response.ok) {
      discoverStatus.textContent = payload.error || "Could not add that model.";
      button.disabled = false;
      return;
    }
    discoverStatus.textContent = `Added ${payload.added.name}. Download it from the list above.`;
    await loadModels();
    await runDiscover(discoverQ.value, true);
  } catch (error) {
    discoverStatus.textContent = "Could not reach the local server.";
    button.disabled = false;
  }
}

async function runDiscover(query, quiet) {
  if (discoverSubmit) discoverSubmit.disabled = true;
  if (!quiet) discoverStatus.textContent = query ? `Searching for "${query}"…` : "Searching…";
  try {
    const params = new URLSearchParams({ task: "chat" });
    if (query) params.set("q", query);
    const response = await fetch(`/api/models/discover?${params.toString()}`);
    const payload = await response.json();
    if (!response.ok) {
      discoverStatus.textContent = payload.error || "Search failed.";
      return;
    }
    discoverStatus.textContent = discoverStatusText(payload);
    discoverList.replaceChildren(...(payload.results || []).map(discoverItem));
  } catch (error) {
    discoverStatus.textContent = "Could not reach the local server.";
  } finally {
    if (discoverSubmit) discoverSubmit.disabled = false;
  }
}

if (discoverForm) {
  discoverForm.addEventListener("submit", (event) => {
    event.preventDefault();
    runDiscover(discoverQ.value.trim(), false);
  });
}
