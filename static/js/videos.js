/* Download library on /videos: check the remote manifest, show what is new
 * or updated, and download selected files with progress. Playback of files
 * already on disk does not go online.
 */
(function () {
  const root = document.getElementById("remote-library");
  if (!root) return;

  const button = document.getElementById("download-library");
  const panel = document.getElementById("library-panel");
  const list = document.getElementById("library-list");
  const summary = document.getElementById("library-summary");
  const checked = document.getElementById("library-checked");
  const errorEl = document.getElementById("library-error");
  const form = document.getElementById("library-form");
  const downloadAll = document.getElementById("library-download-all");
  const progressEl = document.getElementById("library-progress");
  const bar = document.getElementById("library-bar");
  const barFill = bar ? bar.querySelector("span") : null;

  let pollTimer = null;

  function showError(message) {
    if (!errorEl) return;
    if (message) {
      errorEl.textContent = message;
      errorEl.hidden = false;
    } else {
      errorEl.textContent = "";
      errorEl.hidden = true;
    }
  }

  function setChecked(state) {
    if (!checked) return;
    if (state && state.last_checked) {
      const flag = state.last_checked_ok ? "ok" : "failed";
      checked.textContent = "Remote manifest last checked " + state.last_checked + " (" + flag + ").";
    } else {
      checked.textContent = "The remote manifest has not been checked yet.";
    }
  }

  function renderComparison(comparison) {
    panel.hidden = false;
    list.innerHTML = "";
    if (!comparison || !comparison.videos || !comparison.videos.length) {
      summary.textContent = "The remote library is empty, or the manifest has not been checked.";
      return;
    }
    const c = comparison.counts || {};
    const pending = (c.new || 0) + (c.updated || 0);
    if (pending === 0) {
      summary.textContent = "Everything in the remote library is already on disk (" + comparison.total + " video(s)).";
    } else {
      summary.textContent =
        (c.new || 0) + " new, " + (c.updated || 0) + " updated, " +
        (c.current || 0) + " already on disk. To download: " + (comparison.download_label || "") + ".";
    }
    comparison.videos.forEach(function (item) {
      const li = document.createElement("li");
      const label = document.createElement("label");
      const box = document.createElement("input");
      box.type = "checkbox";
      box.name = "id";
      box.value = item.id;
      box.checked = item.status === "new" || item.status === "updated";
      box.disabled = item.status === "current";
      const text = document.createElement("span");
      const tag = item.status === "current" ? "on disk" : item.status;
      text.textContent = " " + item.title + " · " + tag + " · " + (item.size_label || "");
      label.appendChild(box);
      label.appendChild(text);
      li.appendChild(label);
      list.appendChild(li);
    });
  }

  function selectedIds() {
    return Array.prototype.map.call(
      list.querySelectorAll('input[type="checkbox"]:checked:not(:disabled)'),
      function (el) { return el.value; }
    );
  }

  function showProgress(job) {
    if (!job || job.state === "idle") {
      progressEl.hidden = true;
      bar.hidden = true;
      return;
    }
    progressEl.hidden = false;
    bar.hidden = false;
    let text = job.message || job.state;
    if (job.bytes_label) text += " — " + job.bytes_label;
    if (job.percent != null) text += " (" + job.percent + "%)";
    progressEl.textContent = text;
    barFill.style.width = (job.percent || 0) + "%";
    if (job.state === "error") showError(job.error || job.message);
  }

  function pollStatus() {
    fetch("/api/videos/library/status")
      .then(function (r) { return r.json(); })
      .then(function (job) {
        showProgress(job);
        if (job.state === "downloading" || job.state === "checking") {
          pollTimer = setTimeout(pollStatus, 400);
        } else {
          pollTimer = null;
          button.disabled = false;
          if (job.state === "done") {
            fetch("/api/videos/library")
              .then(function (r) { return r.json(); })
              .then(function (body) {
                if (body.comparison) renderComparison(body.comparison);
              });
          }
        }
      })
      .catch(function () { pollTimer = setTimeout(pollStatus, 1000); });
  }

  function startDownload(payload) {
    showError("");
    button.disabled = true;
    fetch("/api/videos/library/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(function (r) { return r.json().then(function (body) { return { ok: r.ok, body: body }; }); })
      .then(function (res) {
        if (!res.body.ok) {
          showError(res.body.error || "Could not start the download.");
          button.disabled = false;
        }
        if (res.body.job) showProgress(res.body.job);
        if (res.body.ok || (res.body.job && res.body.job.state === "downloading")) {
          pollStatus();
        } else {
          button.disabled = false;
        }
      })
      .catch(function (err) {
        showError(err.message || "Could not start the download.");
        button.disabled = false;
      });
  }

  button.addEventListener("click", function () {
    showError("");
    button.disabled = true;
    button.textContent = "Checking…";
    fetch("/api/videos/library/check", { method: "POST" })
      .then(function (r) { return r.json().then(function (body) { return { ok: r.ok, body: body }; }); })
      .then(function (res) {
        button.textContent = "Download library";
        button.disabled = false;
        setChecked(res.body);
        if (res.body.comparison) renderComparison(res.body.comparison);
        else panel.hidden = false;
        if (!res.body.ok) {
          showError(res.body.error || "Could not reach the remote library. Anything already on disk is still available.");
        }
      })
      .catch(function (err) {
        button.textContent = "Download library";
        button.disabled = false;
        showError((err && err.message) || "Could not reach the remote library. Anything already on disk is still available.");
        panel.hidden = false;
      });
  });

  form.addEventListener("submit", function (ev) {
    ev.preventDefault();
    const ids = selectedIds();
    if (!ids.length) {
      showError("Select at least one video to download.");
      return;
    }
    startDownload({ ids: ids });
  });

  downloadAll.addEventListener("click", function () {
    startDownload({ all: true });
  });

  // If a download is already running (page reload), keep showing it.
  fetch("/api/videos/library/status")
    .then(function (r) { return r.json(); })
    .then(function (job) {
      if (job && (job.state === "downloading" || job.state === "checking")) {
        panel.hidden = false;
        button.disabled = true;
        showProgress(job);
        pollStatus();
      } else if (job && job.state === "done" && job.files_done) {
        panel.hidden = false;
        showProgress(job);
        fetch("/api/videos/library")
          .then(function (r) { return r.json(); })
          .then(function (body) {
            if (body.comparison) renderComparison(body.comparison);
          });
      }
    })
    .catch(function () { /* stay offline-quiet */ });
})();
