// Model picker: lists /api/models and switches with POST /api/models/select.
// Relies on loadInfo() from app.js to refresh the form after a switch.
const pickerList = document.querySelector("#picker-list");
const pickerExcluded = document.querySelector("#picker-excluded");
const pickerExcludedWrap = document.querySelector("#picker-excluded-wrap");
const pickerCaps = document.querySelector("#picker-caps");
const pickerStatus = document.querySelector("#picker-status");
let pickerBusy = false;

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
  badge.textContent = model.active ? "active" : model.installed ? "on disk" : `download ~${model.approx_size_mb} MB`;
  head.appendChild(badge);
  item.appendChild(head);

  const desc = document.createElement("p");
  desc.className = "extra";
  desc.textContent = `${model.task} · ${model.backend} — ${model.description || ""}`;
  item.appendChild(desc);

  if (model.reasons && model.reasons.length) {
    const why = document.createElement("p");
    why.className = "extra";
    why.textContent = model.reasons.join("; ");
    item.appendChild(why);
  }
  if (selectable && !model.active) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = model.installed ? "Use this model" : "Download and use";
    button.addEventListener("click", () => selectModel(model, button));
    item.appendChild(button);
  }
  return item;
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
  if (payload.weights_override) {
    pickerStatus.textContent = "MODEL_WEIGHTS is set, so switching is turned off.";
  }
}

async function selectModel(model, button) {
  if (pickerBusy) return;
  pickerBusy = true;
  for (const b of document.querySelectorAll("#model-picker button")) b.disabled = true;
  pickerStatus.textContent = model.installed
    ? `Loading ${model.name}…`
    : `Downloading ${model.name}. This needs the internet once.`;
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
