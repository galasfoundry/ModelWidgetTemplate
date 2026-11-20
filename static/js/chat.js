/* Offline chat for /chat. Streams POST /api/chat as Server-Sent Event lines. */
(function () {
  const panel = document.getElementById("chat-panel");
  if (!panel) return;
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send");
  const stopBtn = document.getElementById("chat-stop");
  const clearBtn = document.getElementById("chat-clear");
  const transcript = document.getElementById("transcript");
  const ready = panel.dataset.ready === "true";
  let messages = [];
  let abort = null;

  function addLine(role, text) {
    const li = document.createElement("li");
    li.className = "chat-msg chat-msg-" + role;
    const who = document.createElement("strong");
    who.textContent = role === "user" ? "You" : "The First Book";
    const body = document.createElement("div");
    body.className = "chat-text";
    body.textContent = text;
    li.appendChild(who);
    li.appendChild(body);
    transcript.appendChild(li);
    li.scrollIntoView({ block: "end" });
    return body;
  }

  function setBusy(busy) {
    sendBtn.disabled = !ready || busy;
    input.disabled = !ready || busy;
    stopBtn.disabled = !busy;
    clearBtn.disabled = !ready || busy;
  }

  async function send(event) {
    event.preventDefault();
    if (!ready || sendBtn.disabled) return;
    const text = (input.value || "").trim();
    if (!text) return;
    input.value = "";
    messages.push({ role: "user", content: text });
    addLine("user", text);
    const replyEl = addLine("assistant", "");
    setBusy(true);
    abort = new AbortController();
    let reply = "";
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: messages }),
        signal: abort.signal,
      });
      if (!res.ok && (res.headers.get("Content-Type") || "").indexOf("text/event-stream") === -1) {
        const data = await res.json().catch(function () { return {}; });
        throw new Error(data.error || ("Chat failed (" + res.status + ")"));
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const chunk = await reader.read();
        if (chunk.done) break;
        buf += decoder.decode(chunk.value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() || "";
        for (const part of parts) {
          const line = part.split("\n").find(function (l) { return l.indexOf("data: ") === 0; });
          if (!line) continue;
          const data = JSON.parse(line.slice(6));
          if (data.error) throw new Error(data.error);
          if (data.token) {
            reply += data.token;
            replyEl.textContent = reply;
            replyEl.parentElement.scrollIntoView({ block: "end" });
          }
        }
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        reply = reply || "";
        const note = (reply ? "\n\n" : "") + "Error: " + err.message;
        replyEl.textContent = reply + note;
      }
    } finally {
      if (reply) messages.push({ role: "assistant", content: reply });
      else if (messages.length && messages[messages.length - 1].role === "user") {
        /* keep the user turn so they can try again after a failure */
      }
      setBusy(false);
      abort = null;
      input.focus();
    }
  }

  async function stop() {
    try { await fetch("/api/chat/stop", { method: "POST" }); } catch (e) {}
    if (abort) abort.abort();
  }

  function clear() {
    messages = [];
    transcript.innerHTML = "";
  }

  form.addEventListener("submit", send);
  stopBtn.addEventListener("click", stop);
  clearBtn.addEventListener("click", clear);
})();
