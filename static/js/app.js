const form = document.querySelector("#predict-form");
const imageFields = document.querySelector("#image-fields");
const textFields = document.querySelector("#text-fields");
const imageInput = document.querySelector("#image");
const textInput = document.querySelector("#text");
const preview = document.querySelector("#preview");
const meta = document.querySelector("#meta");
const statusLine = document.querySelector("#status");
const results = document.querySelector("#results");
const submit = document.querySelector("#submit");
const title = document.querySelector("#title");

let inputKind = "image";
let task = "";

function setStatus(message) {
  statusLine.textContent = message;
}

function showEmpty() {
  results.replaceChildren();
  const empty = document.createElement("p");
  empty.className = "empty";
  empty.textContent = "Results will show up here.";
  results.appendChild(empty);
}

async function loadInfo() {
  const response = await fetch("/api/info");
  if (!response.ok) {
    setStatus("Could not read the model configuration.");
    return;
  }
  const info = await response.json();
  inputKind = info.input_kind;
  task = info.task;
  title.textContent = info.model_id;
  const revision = info.revision ? info.revision.slice(0, 12) : "unpinned";
  meta.textContent = `${info.task} · ${info.backend} · ${info.device} · ${info.dtype} · ${revision}`;
  const imageMode = inputKind === "image";
  imageFields.hidden = !imageMode;
  textFields.hidden = imageMode;
  imageInput.required = imageMode;
  textInput.required = !imageMode;
  if (task === "text-generation") {
    textInput.placeholder = "Once upon a time";
    submit.textContent = "Generate";
  } else if (imageMode) {
    submit.textContent = "Classify image";
  } else {
    textInput.placeholder = "A short sentence for the classifier";
    submit.textContent = "Classify text";
  }
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files && imageInput.files[0];
  if (!file) {
    preview.hidden = true;
    preview.removeAttribute("src");
    return;
  }
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
});

function renderResults(payload) {
  results.replaceChildren();
  if (payload.text && task === "text-generation") {
    const block = document.createElement("p");
    block.className = "generated";
    block.textContent = payload.text;
    results.appendChild(block);
    return;
  }
  const items = payload.results || [];
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "The model returned no rows.";
    results.appendChild(empty);
    return;
  }
  const list = document.createElement("ol");
  list.className = "scores";
  for (const row of items) {
    const item = document.createElement("li");
    const name = document.createElement("span");
    name.className = "label";
    name.textContent = row.label || "result";
    item.appendChild(name);
    if (typeof row.score === "number") {
      const pct = Math.max(0, Math.min(100, row.score * 100));
      const bar = document.createElement("span");
      bar.className = "bar";
      const fill = document.createElement("span");
      fill.style.width = `${pct}%`;
      bar.appendChild(fill);
      const score = document.createElement("span");
      score.className = "score";
      score.textContent = `${pct.toFixed(1)}%`;
      item.append(bar, score);
    }
    if (row.description) {
      const desc = document.createElement("p");
      desc.className = "extra";
      desc.textContent = row.description;
      item.appendChild(desc);
    }
    list.appendChild(item);
  }
  results.appendChild(list);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submit.disabled = true;
  setStatus("Running. The first request loads the weights.");
  results.replaceChildren();
  try {
    let response;
    if (inputKind === "image") {
      const file = imageInput.files && imageInput.files[0];
      if (!file) {
        setStatus("Choose an image first.");
        showEmpty();
        return;
      }
      const body = new FormData();
      body.append("image", file);
      response = await fetch("/api/predict", { method: "POST", body });
    } else {
      response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textInput.value }),
      });
    }
    const payload = await response.json();
    if (!response.ok) {
      setStatus(payload.error || "The request failed.");
      showEmpty();
      return;
    }
    setStatus("Done.");
    renderResults(payload);
  } catch (error) {
    setStatus("Could not reach the local server.");
    showEmpty();
  } finally {
    submit.disabled = false;
  }
});

loadInfo();
