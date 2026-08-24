/* Local model chat UI. One instance per model profile; capabilities such as
 * image input and the thinking block come from /api/config. */
(function () {
  "use strict";

  const STORAGE_KEY = "chat-conversations";
  const SETTINGS_KEY = "chat-settings";
  const THEME_KEY = "chat-theme";
  const SESSION_ID_KEY = "chat-session-id";

  // Images are re-encoded below this edge length so a screenshot cannot blow
  // up the context, and previewed far smaller so the tab-scoped history stays bounded.
  const MAX_IMAGE_EDGE = 2048;
  const PREVIEW_EDGE = 480;

  const DEFAULT_SETTINGS = {
    system: "",
    temperature: 0.7,
    top_p: 0.95,
    top_k: 64,
    max_tokens: 2048,
    stripDx: true,
    thinking: true,
  };

  const SUGGESTIONS = {
    medgemma: [
      "QTc 用 Bazett 和 Fridericia 校正有什么区别？分别适合什么心率范围？",
      "左束支传导阻滞存在时，如何判断急性心肌梗死？",
      "请解释 ECG 上区分室性心动过速与差异性传导室上速的要点。",
    ],
    default: [
      "解释一下 ECG 电轴左偏的常见原因和判断方法。",
      "帮我读一张心电图：我把图片传给你，请先描述看到的波形再下结论。",
      "用 Python 写一个函数，从 R 波位置序列计算 HRV 的 SDNN 和 RMSSD。",
    ],
  };

  const $ = (id) => document.getElementById(id);
  const el = {
    messages: $("messages"),
    emptyState: $("empty-state"),
    input: $("input"),
    send: $("send"),
    stop: $("stop"),
    convList: $("conv-list"),
    convTitle: $("conv-title"),
    attachments: $("attachments"),
    fileInput: $("file-input"),
    stats: $("stats"),
    statusDot: $("status-dot"),
    statusText: $("status-text"),
    toast: $("toast"),
    modal: $("settings-modal"),
    sidebar: $("sidebar"),
  };

  let settings = Object.assign({}, DEFAULT_SETTINGS);
  let serverConfig = { label: "本地模型", supports_images: false, supports_thinking: false };
  let conversations = [];
  let currentId = null;
  let pending = [];        // attachments staged for the next message
  let controller = null;   // aborts the in-flight generation
  let apiToken = "";       // deliberately memory-only; refresh requires re-entry
  let chatSessionId = sessionStorage.getItem(SESSION_ID_KEY) || "";
  if (!/^[A-Za-z0-9_-]{16,80}$/.test(chatSessionId)) {
    chatSessionId = (self.crypto && crypto.randomUUID)
      ? crypto.randomUUID().replace(/-/g, "")
      : Array.from(crypto.getRandomValues(new Uint8Array(24)), (byte) => byte.toString(16).padStart(2, "0")).join("");
    sessionStorage.setItem(SESSION_ID_KEY, chatSessionId);
  }

  /* ---------- persistence ---------- */

  function storageKey(base) {
    // Each model profile keeps its own history and settings.
    return base + ":" + (serverConfig.profile || "default");
  }

  function load(key, fallback) {
    try {
      const raw = sessionStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch (err) {
      return fallback;
    }
  }

  function saveConversations() {
    try {
      sessionStorage.setItem(storageKey(STORAGE_KEY), JSON.stringify(conversations));
    } catch (err) {
      toast("当前标签页的对话历史保存失败（浏览器存储已满）");
    }
  }

  function saveSettings() {
    sessionStorage.setItem(storageKey(SETTINGS_KEY), JSON.stringify(settings));
  }

  async function apiFetch(url, options, retryAuth) {
    const init = Object.assign({}, options || {});
    const headers = new Headers(init.headers || {});
    headers.set("X-Chat-Session", chatSessionId);
    if (apiToken) headers.set("X-Chat-Token", apiToken);
    init.headers = headers;
    const response = await fetch(url, init);
    if (response.status === 401 && retryAuth !== false) {
      const supplied = window.prompt("此聊天服务需要访问令牌。令牌仅保存在当前页面内存中：", "");
      if (supplied) {
        apiToken = supplied;
        return apiFetch(url, options, false);
      }
    }
    return response;
  }

  function attachmentsInMessages(messages) {
    const ids = new Set();
    (messages || []).forEach((message) => {
      (message.attachments || []).forEach((file) => {
        if (file.id) ids.add(file.id);
      });
    });
    return Array.from(ids);
  }

  function deleteRemoteAttachments(messages) {
    attachmentsInMessages(messages).forEach((id) => {
      apiFetch("/api/attachments/" + id, { method: "DELETE" }).catch(() => {});
    });
  }

  function discardPending() {
    deleteRemoteAttachments([{ attachments: pending }]);
    pending = [];
  }

  function clearMedicalBrowserData() {
    [localStorage, sessionStorage].forEach((storage) => {
      for (let index = storage.length - 1; index >= 0; index -= 1) {
        const key = storage.key(index) || "";
        if (key.startsWith(STORAGE_KEY + ":") || key.startsWith(SETTINGS_KEY + ":")) {
          storage.removeItem(key);
        }
      }
    });
  }

  function current() {
    return conversations.find((conv) => conv.id === currentId) || null;
  }

  function newConversation() {
    discardPending();
    const conv = {
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 7),
      title: "新的对话",
      updatedAt: Date.now(),
      messages: [],
    };
    conversations.unshift(conv);
    currentId = conv.id;
    saveConversations();
    renderConvList();
    renderMessages();
    renderPending();
    el.input.focus();
  }

  /* ---------- rendering ---------- */

  function toast(text) {
    el.toast.textContent = text;
    el.toast.hidden = false;
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => (el.toast.hidden = true), 2600);
  }

  function renderConvList() {
    el.convList.innerHTML = "";
    conversations.forEach((conv) => {
      const item = document.createElement("div");
      item.className = "conv-item" + (conv.id === currentId ? " active" : "");
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = conv.title;
      const del = document.createElement("button");
      del.className = "del";
      del.textContent = "✕";
      del.title = "删除对话";
      del.onclick = (event) => {
        event.stopPropagation();
        deleteRemoteAttachments(conv.messages);
        conversations = conversations.filter((row) => row.id !== conv.id);
        if (currentId === conv.id) {
          if (conversations.length) currentId = conversations[0].id;
          else return newConversation();
        }
        saveConversations();
        renderConvList();
        renderMessages();
      };
      item.append(name, del);
      item.onclick = () => {
        currentId = conv.id;
        discardPending();
        renderConvList();
        renderMessages();
        renderPending();
        el.sidebar.classList.remove("open");
      };
      el.convList.appendChild(item);
    });
  }

  function fileChip(file, onRemove) {
    const chip = document.createElement("span");
    chip.className = "file-chip" + (file.pending ? " pending" : "");
    if (file.kind === "image" && file.preview) {
      const thumb = document.createElement("img");
      thumb.className = "thumb";
      thumb.src = file.preview;
      chip.appendChild(thumb);
    } else {
      const icon = document.createElement("span");
      icon.textContent =
        file.kind === "ecg_features" ? "🫀" : file.kind === "ecg_report" ? "📄" : file.kind === "image" ? "🖼" : "📎";
      chip.appendChild(icon);
    }
    const name = document.createElement("span");
    name.className = "fname";
    name.textContent = file.name;
    name.title = file.chars ? file.name + " · " + file.chars + " 字符" : file.name;
    chip.appendChild(name);
    if (onRemove) {
      const remove = document.createElement("button");
      remove.className = "rm";
      remove.textContent = "✕";
      remove.onclick = onRemove;
      chip.appendChild(remove);
    }
    return chip;
  }

  function renderPending() {
    el.attachments.innerHTML = "";
    pending.forEach((file, index) => {
      el.attachments.appendChild(
        fileChip(file, () => {
          if (file.id) apiFetch("/api/attachments/" + file.id, { method: "DELETE" });
          pending.splice(index, 1);
          renderPending();
        })
      );
    });
  }

  function thinkingSummary(message, live) {
    if (live) return "思考中…";
    const chars = (message.thinking || "").length;
    return "思考过程 · " + chars + " 字（点击展开）";
  }

  function messageNode(message, index) {
    const wrap = document.createElement("div");
    wrap.className = "msg " + message.role;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = message.role === "user" ? "你" : serverConfig.icon || "🤖";

    const body = document.createElement("div");
    body.className = "body";

    const files = (message.attachments || []).filter((file) => file.kind !== "image");
    const images = (message.attachments || []).filter((file) => file.kind === "image");

    if (images.length) {
      const gallery = document.createElement("div");
      gallery.className = "msg-images";
      images.forEach((file) => {
        if (!file.preview) return;
        const img = document.createElement("img");
        img.src = file.preview;
        img.alt = file.name;
        img.title = file.name;
        img.onclick = () => lightbox(file.preview);
        gallery.appendChild(img);
      });
      body.appendChild(gallery);
    }
    if (files.length) {
      const row = document.createElement("div");
      row.className = "msg-files";
      files.forEach((file) => row.appendChild(fileChip(file, null)));
      body.appendChild(row);
    }

    if (message.role === "assistant") {
      const think = document.createElement("details");
      think.className = "thinking";
      think.hidden = !(message.thinking || "").trim();
      const summary = document.createElement("summary");
      summary.textContent = thinkingSummary(message, false);
      const thinkBody = document.createElement("div");
      thinkBody.className = "think-body";
      thinkBody.textContent = message.thinking || "";
      think.append(summary, thinkBody);
      body.appendChild(think);
    }

    const bubble = document.createElement("div");
    bubble.className = "bubble" + (message.role === "assistant" ? " md" : "");
    if (message.role === "assistant") bubble.innerHTML = window.renderMarkdown(message.content);
    else bubble.textContent = message.content;
    body.appendChild(bubble);

    if (message.error) {
      const error = document.createElement("div");
      error.className = "err";
      error.textContent = message.error;
      body.appendChild(error);
    }

    if (message.meta) {
      const meta = document.createElement("div");
      meta.className = "turn-meta";
      meta.textContent = message.meta;
      body.appendChild(meta);
    }

    const actions = document.createElement("div");
    actions.className = "msg-actions";
    const copy = document.createElement("button");
    copy.textContent = "复制";
    copy.onclick = () => {
      navigator.clipboard.writeText(message.content).then(() => toast("已复制"));
    };
    actions.appendChild(copy);
    if (message.role === "assistant") {
      const again = document.createElement("button");
      again.textContent = "重新生成";
      again.onclick = () => regenerate(index);
      actions.appendChild(again);
    }
    if (message.role === "user") {
      const edit = document.createElement("button");
      edit.textContent = "编辑";
      edit.onclick = () => {
        const conv = current();
        el.input.value = message.content;
        deleteRemoteAttachments(conv.messages.slice(index));
        conv.messages = conv.messages.slice(0, index);
        saveConversations();
        renderMessages();
        autoGrow();
        el.input.focus();
      };
      actions.appendChild(edit);
    }
    body.appendChild(actions);

    wrap.append(avatar, body);
    return wrap;
  }

  function lightbox(src) {
    const layer = document.createElement("div");
    layer.className = "lightbox";
    const img = document.createElement("img");
    img.src = src;
    layer.appendChild(img);
    layer.onclick = () => layer.remove();
    document.body.appendChild(layer);
  }

  function renderMessages() {
    const conv = current();
    el.convTitle.textContent = conv ? conv.title : "新的对话";
    el.messages.innerHTML = "";
    if (!conv || !conv.messages.length) {
      el.messages.appendChild(el.emptyState);
      el.emptyState.hidden = false;
      return;
    }
    conv.messages.forEach((message, index) => el.messages.appendChild(messageNode(message, index)));
    scrollToBottom(true);
  }

  function scrollToBottom(force) {
    const gap = el.messages.scrollHeight - el.messages.scrollTop - el.messages.clientHeight;
    if (force || gap < 160) el.messages.scrollTop = el.messages.scrollHeight;
  }

  /* ---------- generation ---------- */

  function setBusy(busy) {
    el.send.hidden = busy;
    el.stop.hidden = !busy;
  }

  async function streamAssistant(conv) {
    const payload = {
      messages: conv.messages
        .filter((message) => !message.error || message.content)
        .map((message) => ({
          role: message.role,
          content: message.content,
          attachments: (message.attachments || []).map((file) => file.id).filter(Boolean),
        })),
      system: settings.system || serverConfig.default_system_prompt || "",
      temperature: Number(settings.temperature),
      top_p: Number(settings.top_p),
      top_k: Number(settings.top_k),
      max_tokens: Number(settings.max_tokens),
      thinking: Boolean(settings.thinking),
    };

    const target = { role: "assistant", content: "", thinking: "", meta: "" };
    conv.messages.push(target);
    const node = messageNode(target, conv.messages.length - 1);
    const bubble = node.querySelector(".bubble");
    const think = node.querySelector(".thinking");
    const thinkSummary = node.querySelector(".thinking > summary");
    const thinkBody = node.querySelector(".think-body");
    el.emptyState.hidden = true;
    if (el.emptyState.parentNode === el.messages) el.messages.removeChild(el.emptyState);
    el.messages.appendChild(node);
    bubble.innerHTML = '<span class="cursor"></span>';
    scrollToBottom(true);

    controller = new AbortController();
    setBusy(true);
    el.stats.textContent = "生成中…";

    let dirty = false;
    let thinkDirty = false;
    const finishThinking = () => {
      if (!think.hidden && think.classList.contains("live")) {
        think.classList.remove("live");
        think.open = false;
        thinkSummary.textContent = thinkingSummary(target, false);
      }
    };
    const paint = () => {
      if (thinkDirty) {
        thinkBody.textContent = target.thinking;
        thinkBody.scrollTop = thinkBody.scrollHeight;
        thinkDirty = false;
        scrollToBottom(false);
      }
      if (dirty) {
        bubble.innerHTML = window.renderMarkdown(target.content) + '<span class="cursor"></span>';
        dirty = false;
        scrollToBottom(false);
      }
      if (controller) requestAnimationFrame(paint);
    };
    requestAnimationFrame(paint);

    try {
      const response = await apiFetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) throw new Error("HTTP " + response.status);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() || "";
        for (const frame of frames) {
          const line = frame.split("\n").find((row) => row.startsWith("data:"));
          if (!line) continue;
          let event;
          try {
            event = JSON.parse(line.slice(5).trim());
          } catch (err) {
            continue;
          }
          if (event.type === "delta") {
            finishThinking();
            target.content += event.text;
            dirty = true;
          } else if (event.type === "thinking") {
            if (think.hidden) {
              think.hidden = false;
              think.open = true;
              think.classList.add("live");
              thinkSummary.textContent = thinkingSummary(target, true);
            }
            target.thinking += event.text;
            thinkDirty = true;
          } else if (event.type === "error") {
            target.error = event.error;
          } else if (event.type === "done") {
            const bits = [];
            if (event.prompt_tokens != null) bits.push("输入 " + event.prompt_tokens + " tok");
            if (event.completion_tokens != null) bits.push("输出 " + event.completion_tokens + " tok");
            if (event.tokens_per_s) bits.push(event.tokens_per_s + " tok/s");
            if (event.ttft_s != null) bits.push("首字 " + event.ttft_s + "s");
            bits.push("共 " + event.elapsed_s + "s");
            target.meta = bits.join(" · ");
          }
        }
      }
    } catch (err) {
      if (err.name === "AbortError") target.meta = (target.meta || "") + " · 已手动停止";
      else target.error = "请求失败：" + err.message;
    } finally {
      finishThinking();
      controller = null;
      setBusy(false);
      el.stats.textContent = target.meta || "";
      conv.updatedAt = Date.now();
      saveConversations();
      renderMessages();
    }
  }

  async function send() {
    const text = el.input.value.trim();
    if (!text && !pending.length) return;
    if (controller) return;
    if (pending.some((file) => file.pending)) return toast("附件还在上传中…");

    let conv = current();
    if (!conv) {
      newConversation();
      conv = current();
    }

    conv.messages.push({
      role: "user",
      content: text,
      attachments: pending.map((file) => ({
        id: file.id,
        name: file.name,
        kind: file.kind,
        chars: file.chars,
        preview: file.preview,
      })),
    });
    if (conv.messages.length === 1 || conv.title === "新的对话") {
      conv.title = (text || pending[0].name).slice(0, 28);
      el.convTitle.textContent = conv.title;
    }
    el.input.value = "";
    pending = [];
    renderPending();
    autoGrow();
    renderConvList();
    renderMessages();
    saveConversations();

    await streamAssistant(conv);
  }

  async function regenerate(index) {
    if (controller) return;
    const conv = current();
    conv.messages = conv.messages.slice(0, index);
    renderMessages();
    await streamAssistant(conv);
  }

  /* ---------- attachments ---------- */

  function loadImage(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        resolve(img);
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error("无法读取图片"));
      };
      img.src = url;
    });
  }

  function drawScaled(img, maxEdge, type, quality) {
    const scale = Math.min(1, maxEdge / Math.max(img.width, img.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(img.width * scale));
    canvas.height = Math.max(1, Math.round(img.height * scale));
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL(type, quality);
  }

  function dataUrlToBlob(dataUrl) {
    const [head, body] = dataUrl.split(",");
    const mime = head.match(/data:([^;]+)/)[1];
    const bytes = atob(body);
    const buffer = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i += 1) buffer[i] = bytes.charCodeAt(i);
    return new Blob([buffer], { type: mime });
  }

  async function prepareImage(file) {
    const img = await loadImage(file);
    // Keep PNG for PNG sources: JPEG artifacts smear thin ECG traces.
    const type = file.type === "image/png" ? "image/png" : "image/jpeg";
    const preview = drawScaled(img, PREVIEW_EDGE, "image/jpeg", 0.7);
    const oversized = Math.max(img.width, img.height) > MAX_IMAGE_EDGE;
    const upload = oversized ? dataUrlToBlob(drawScaled(img, MAX_IMAGE_EDGE, type, 0.92)) : file;
    return { preview, upload, resized: oversized, width: img.width, height: img.height };
  }

  async function uploadFiles(files) {
    for (const file of files) {
      const isImage = /^image\//.test(file.type);
      if (isImage && !serverConfig.supports_images) {
        toast(serverConfig.label + " 是纯文本模型，无法读取图片");
        continue;
      }

      const placeholder = { name: file.name, kind: isImage ? "image" : "text", pending: true };
      pending.push(placeholder);
      renderPending();

      let payload = file;
      try {
        if (isImage) {
          const prepared = await prepareImage(file);
          placeholder.preview = prepared.preview;
          payload = prepared.upload;
          if (prepared.resized) {
            toast("图片较大，已缩放到 " + MAX_IMAGE_EDGE + "px 以内再送入模型");
          }
          renderPending();
        }
      } catch (err) {
        pending = pending.filter((row) => row !== placeholder);
        renderPending();
        toast(String(err.message || err));
        continue;
      }

      const form = new FormData();
      form.append("file", payload, file.name);
      form.append("strip_dx", settings.stripDx ? "true" : "false");
      try {
        const response = await apiFetch("/api/upload", { method: "POST", body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "上传失败");
        Object.assign(placeholder, data, { pending: false, preview: placeholder.preview });
        if (data.kind === "ecg_features") toast("已识别为 ECG 特征文件，将按流水线格式摘要后送入模型");
        if (data.truncated) toast("文件较长，已截断后送入模型");
      } catch (err) {
        pending = pending.filter((row) => row !== placeholder);
        toast(String(err.message || err));
      }
      renderPending();
    }
  }

  /* ---------- settings ---------- */

  function syncSettingsInputs() {
    $("set-system").value = settings.system || serverConfig.default_system_prompt || "";
    $("set-temp").value = settings.temperature;
    $("set-topp").value = settings.top_p;
    $("set-topk").value = settings.top_k;
    $("set-maxtok").value = settings.max_tokens;
    $("set-stripdx").checked = settings.stripDx;
    $("set-thinking").checked = settings.thinking;
    $("thinking-field").hidden = !serverConfig.supports_thinking;
    $("val-temp").textContent = Number(settings.temperature).toFixed(2);
    $("val-topp").textContent = Number(settings.top_p).toFixed(2);
    $("val-topk").textContent = settings.top_k;
  }

  function bindSettings() {
    $("open-settings").onclick = () => {
      syncSettingsInputs();
      el.modal.hidden = false;
    };
    const close = () => {
      settings.system = $("set-system").value;
      settings.temperature = Number($("set-temp").value);
      settings.top_p = Number($("set-topp").value);
      settings.top_k = Number($("set-topk").value);
      settings.max_tokens = Number($("set-maxtok").value);
      settings.stripDx = $("set-stripdx").checked;
      settings.thinking = $("set-thinking").checked;
      saveSettings();
      el.modal.hidden = true;
    };
    $("close-settings").onclick = close;
    $("save-settings").onclick = close;
    $("reset-settings").onclick = () => {
      settings = Object.assign({}, DEFAULT_SETTINGS, {
        system: serverConfig.default_system_prompt || "",
        max_tokens: serverConfig.default_max_tokens || DEFAULT_SETTINGS.max_tokens,
      });
      saveSettings();
      syncSettingsInputs();
    };
    $("clear-local-data").onclick = () => {
      deleteRemoteAttachments(conversations.flatMap((conv) => conv.messages || []));
      discardPending();
      clearMedicalBrowserData();
      conversations = [];
      currentId = null;
      settings = Object.assign({}, DEFAULT_SETTINGS, {
        system: serverConfig.default_system_prompt || "",
        max_tokens: serverConfig.default_max_tokens || DEFAULT_SETTINGS.max_tokens,
      });
      newConversation();
      syncSettingsInputs();
      toast("已清除当前浏览器中的全部对话数据");
    };
    ["set-temp", "set-topp", "set-topk"].forEach((id) => {
      $(id).oninput = () => {
        const value = Number($(id).value);
        if (id === "set-temp") $("val-temp").textContent = value.toFixed(2);
        if (id === "set-topp") $("val-topp").textContent = value.toFixed(2);
        if (id === "set-topk") $("val-topk").textContent = value;
      };
    });
    el.modal.onclick = (event) => {
      if (event.target === el.modal) close();
    };
  }

  /* ---------- health ---------- */

  async function checkHealth() {
    try {
      const response = await apiFetch("/api/health");
      const data = await response.json();
      if (data.ok) {
        el.statusDot.className = "dot ok";
        const name = String(data.model || "").split("/").filter(Boolean).pop();
        el.statusText.textContent =
          name + (data.max_model_len ? " · " + (data.max_model_len / 1024).toFixed(0) + "k ctx" : "");
        el.statusText.title = data.model + " @ " + data.upstream;
      } else {
        el.statusDot.className = "dot bad";
        el.statusText.textContent = "模型服务未就绪";
        el.statusText.title = data.error || data.upstream;
      }
    } catch (err) {
      el.statusDot.className = "dot bad";
      el.statusText.textContent = "无法连接后端";
    }
  }

  /* ---------- misc UI ---------- */

  function autoGrow() {
    el.input.style.height = "auto";
    el.input.style.height = Math.min(el.input.scrollHeight, 220) + "px";
  }

  function exportChat() {
    const conv = current();
    if (!conv || !conv.messages.length) return toast("当前对话为空");
    const lines = ["# " + conv.title, ""];
    conv.messages.forEach((message) => {
      lines.push(message.role === "user" ? "## 我" : "## " + serverConfig.label);
      if (message.attachments && message.attachments.length) {
        lines.push("附件：" + message.attachments.map((file) => file.name).join("、"));
      }
      lines.push("", message.content, "");
    });
    const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = conv.title.replace(/[^\w一-龥-]+/g, "_") + ".md";
    link.click();
    URL.revokeObjectURL(link.href);
  }

  function applyServerConfig() {
    const label = serverConfig.label || "本地模型";
    const icon = serverConfig.icon || "🤖";
    document.title = label + " 本地对话";
    $("empty-title").textContent = "与本地 " + label + " 对话";
    document.querySelector(".empty-icon").textContent = icon;
    const favicon = document.querySelector("link[rel=icon]");
    if (favicon) {
      favicon.href =
        "data:image/svg+xml," +
        encodeURIComponent(
          `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><text y="26" font-size="26">${icon}</text></svg>`
        );
    }

    const list = SUGGESTIONS[serverConfig.profile] || SUGGESTIONS.default;
    const box = $("suggestions");
    box.innerHTML = "";
    list.forEach((text) => {
      const button = document.createElement("button");
      button.className = "suggestion";
      button.textContent = text;
      box.appendChild(button);
    });

    const hints = ["可以用 📎 上传 *_features.json 或 *_report.txt，模型会按流水线同款格式读取其中的测量值"];
    if (serverConfig.supports_images) hints.push("这个模型能读图，心电图截图可以直接拖进来");
    if (serverConfig.supports_thinking) hints.push("思考过程会折叠显示，可在「生成参数」里关闭");
    $("empty-hint").textContent = "提示：" + hints.join("；") + "。";

    if (Array.isArray(serverConfig.accept)) el.fileInput.accept = serverConfig.accept.join(",");
    $("attach-btn").title = serverConfig.supports_images
      ? "上传 ECG 特征、报告或图片"
      : "上传 ECG 特征或报告";
  }

  function bind() {
    el.send.onclick = send;
    el.stop.onclick = () => controller && controller.abort();
    $("new-chat").onclick = newConversation;
    $("clear-chat").onclick = () => {
      const conv = current();
      if (!conv) return;
      deleteRemoteAttachments(conv.messages);
      discardPending();
      conv.messages = [];
      conv.title = "新的对话";
      saveConversations();
      renderConvList();
      renderMessages();
    };
    $("export-chat").onclick = exportChat;
    $("toggle-sidebar").onclick = () => el.sidebar.classList.toggle("open");
    $("toggle-theme").onclick = () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      localStorage.setItem(THEME_KEY, next);
    };

    el.input.addEventListener("input", autoGrow);
    el.input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        send();
      }
    });

    $("attach-btn").onclick = () => el.fileInput.click();
    el.fileInput.onchange = () => {
      uploadFiles(Array.from(el.fileInput.files));
      el.fileInput.value = "";
    };

    // Paste an image straight from the clipboard.
    el.input.addEventListener("paste", (event) => {
      const files = Array.from(event.clipboardData ? event.clipboardData.files : []);
      if (files.length) {
        event.preventDefault();
        uploadFiles(files);
      }
    });

    document.addEventListener("dragover", (event) => event.preventDefault());
    document.addEventListener("drop", (event) => {
      event.preventDefault();
      if (event.dataTransfer.files.length) uploadFiles(Array.from(event.dataTransfer.files));
    });

    document.addEventListener("click", (event) => {
      if (event.target.classList.contains("suggestion")) {
        el.input.value = event.target.textContent;
        autoGrow();
        send();
      }
      if (event.target.classList.contains("code-copy")) {
        const code = event.target.parentNode.querySelector("code");
        navigator.clipboard.writeText(code.textContent).then(() => toast("代码已复制"));
      }
    });
  }

  async function init() {
    document.documentElement.dataset.theme =
      localStorage.getItem(THEME_KEY) ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

    bind();
    bindSettings();

    // The profile decides which history bucket to read, so load config first.
    try {
      const response = await apiFetch("/api/config");
      if (!response.ok) throw new Error("HTTP " + response.status);
      serverConfig = await response.json();
    } catch (err) {
      /* the health indicator already surfaces backend problems */
    }
    applyServerConfig();

    // Remove data written by older releases. Medical conversation history is
    // now tab-scoped and disappears when the tab closes.
    localStorage.removeItem(storageKey(STORAGE_KEY));
    localStorage.removeItem(storageKey(SETTINGS_KEY));

    settings = Object.assign(
      {},
      DEFAULT_SETTINGS,
      { max_tokens: serverConfig.default_max_tokens || DEFAULT_SETTINGS.max_tokens },
      load(storageKey(SETTINGS_KEY), {})
    );
    if (!settings.system) settings.system = serverConfig.default_system_prompt || "";

    conversations = load(storageKey(STORAGE_KEY), []);
    if (!conversations.length) newConversation();
    else currentId = conversations[0].id;

    renderConvList();
    renderMessages();

    checkHealth();
    setInterval(checkHealth, 20000);
    el.input.focus();
  }

  init();
})();
