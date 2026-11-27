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

/* Branch lanes: watched badges, a quiet theme when a lane is opened, per-episode download. */
(function () {
  const tree = document.getElementById("branch-tree");
  const WATCHED_KEY = "firstbook.watched";
  const SOUND_KEY = "firstbook.branchSound";

  function readWatched() {
    try {
      const raw = JSON.parse(localStorage.getItem(WATCHED_KEY) || "[]");
      return new Set(Array.isArray(raw) ? raw : []);
    } catch (e) {
      return new Set();
    }
  }

  function writeWatched(set) {
    localStorage.setItem(WATCHED_KEY, JSON.stringify(Array.from(set)));
  }

  function markWatched(id) {
    if (!id) return;
    const set = readWatched();
    set.add(id);
    writeWatched(set);
    paintWatched();
  }

  function paintWatched() {
    const set = readWatched();
    document.querySelectorAll(".episode-card[data-watch-id]").forEach(function (card) {
      const badge = card.querySelector(".ep-badge.watched");
      if (!badge) return;
      const on = set.has(card.getAttribute("data-watch-id"));
      badge.hidden = !on;
    });
  }

  function soundOn() {
    return localStorage.getItem(SOUND_KEY) === "on";
  }

  function paintSound() {
    const button = document.getElementById("branch-sound");
    if (!button) return;
    const on = soundOn();
    button.textContent = on ? "Sound on" : "Sound off";
    button.setAttribute("aria-pressed", on ? "false" : "true");
  }

  function themeUrl(lane) {
    const audio = document.createElement("audio");
    const ogg = lane.getAttribute("data-theme-ogg");
    const mp3 = lane.getAttribute("data-theme-mp3");
    if (ogg && audio.canPlayType('audio/ogg; codecs="vorbis"')) return ogg;
    return mp3 || ogg;
  }

  function playLane(lane) {
    const note = document.getElementById("branch-playing");
    const player = document.getElementById("branch-player");
    if (!player || !lane) return;
    if (!soundOn()) {
      if (note) note.hidden = true;
      return;
    }
    const url = themeUrl(lane);
    if (!url) return;
    player.volume = 0.22;
    player.src = url;
    const title = lane.getAttribute("data-title") || "branch";
    if (note) {
      note.hidden = false;
      note.textContent = "Playing the " + title + " theme.";
    }
    player.play().catch(function () {
      if (note) note.hidden = true;
    });
  }

  paintWatched();

  const player = document.getElementById("branch-player");
  if (player) {
    player.addEventListener("ended", function () {
      const note = document.getElementById("branch-playing");
      if (note) note.hidden = true;
    });
  }

  document.querySelectorAll("video[data-watch-id]").forEach(function (video) {
    const id = video.getAttribute("data-watch-id");
    function maybe() {
      if (video.duration && video.currentTime / video.duration >= 0.9) markWatched(id);
    }
    video.addEventListener("ended", function () { markWatched(id); });
    video.addEventListener("timeupdate", maybe);
  });

  if (!tree) {
    const watchButton = document.getElementById("watch-download");
    if (watchButton) watchButton.addEventListener("click", function () { downloadOne(watchButton); });
    return;
  }

  paintSound();
  const born = performance.now();

  document.getElementById("branch-sound").addEventListener("click", function () {
    localStorage.setItem(SOUND_KEY, soundOn() ? "off" : "on");
    paintSound();
    if (soundOn()) {
      const open = tree.querySelector("details[open]");
      if (open) playLane(open.closest(".branch-lane"));
    } else {
      const playing = document.getElementById("branch-player");
      if (playing) playing.pause();
      const note = document.getElementById("branch-playing");
      if (note) note.hidden = true;
    }
  });

  tree.querySelectorAll(".branch-lane details").forEach(function (details) {
    details.addEventListener("toggle", function (ev) {
      if (!ev.isTrusted || performance.now() - born < 400) return;
      if (details.open) playLane(details.closest(".branch-lane"));
    });
  });

  tree.querySelectorAll(".ep-download").forEach(function (button) {
    button.addEventListener("click", function () { downloadOne(button); });
  });

  function showBranchError(message) {
    const errorEl = document.getElementById("library-error") || document.getElementById("watch-download-status");
    if (errorEl) {
      errorEl.textContent = message;
      errorEl.hidden = !message;
      return;
    }
    const note = document.getElementById("branch-playing");
    if (note) {
      note.hidden = !message;
      note.textContent = message;
    }
  }

  function downloadOne(button) {
    const id = button.getAttribute("data-manifest-id");
    if (!id) return;
    button.disabled = true;
    button.textContent = "Downloading…";
    showBranchError("");
    fetch("/api/videos/library/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: [id] }),
    })
      .then(function (r) { return r.json().then(function (body) { return { ok: r.ok, body: body }; }); })
      .then(function (res) {
        if (!res.body.ok) {
          showBranchError(res.body.error || "Could not start the download.");
          button.disabled = false;
          button.textContent = "Download";
          return;
        }
        pollDownload(button);
      })
      .catch(function (err) {
        showBranchError((err && err.message) || "Could not start the download.");
        button.disabled = false;
        button.textContent = "Download";
      });
  }

  function pollDownload(button) {
    fetch("/api/videos/library/status")
      .then(function (r) { return r.json(); })
      .then(function (job) {
        if (job.state === "downloading" || job.state === "checking") {
          setTimeout(function () { pollDownload(button); }, 400);
          return;
        }
        if (job.state === "error") {
          showBranchError(job.error || job.message || "Download failed.");
          button.disabled = false;
          button.textContent = "Download";
          return;
        }
        window.location.reload();
      })
      .catch(function () { setTimeout(function () { pollDownload(button); }, 1000); });
  }

  const watchButton = document.getElementById("watch-download");
  if (watchButton) watchButton.addEventListener("click", function () { downloadOne(watchButton); });
})();
