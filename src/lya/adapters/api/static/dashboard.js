// Lya Dashboard — WebSocket client + REST
(function() {
    "use strict";

    const statusEl = document.getElementById("status");
    const eventLog = document.getElementById("event-log");
    const chatLog = document.getElementById("chat-log");
    const chatInput = document.getElementById("chat-input");
    const chatSend = document.getElementById("chat-send");

    let ws = null;
    let reconnectTimer = null;

    // WebSocket connection
    function connectWS() {
        const proto = location.protocol === "https:" ? "wss:" : "ws:";
        const url = `${proto}//${location.host}/ws`;

        ws = new WebSocket(url);

        ws.onopen = function() {
            statusEl.textContent = "Connected";
            statusEl.className = "status online";
            addEvent("system", "WebSocket connected");
        };

        ws.onmessage = function(evt) {
            try {
                const data = JSON.parse(evt.data);
                handleWSMessage(data);
            } catch(e) {
                addEvent("error", "Invalid message: " + evt.data);
            }
        };

        ws.onclose = function() {
            statusEl.textContent = "Disconnected";
            statusEl.className = "status offline";
            addEvent("system", "WebSocket disconnected, reconnecting...");
            reconnectTimer = setTimeout(connectWS, 3000);
        };

        ws.onerror = function() {
            addEvent("error", "WebSocket error");
        };
    }

    function handleWSMessage(data) {
        const type = data.type || "unknown";

        if (type === "message") {
            addChatMessage(data.username || "User", data.text || "", "user");
            if (data.response) {
                addChatMessage("Lya", data.response, "bot");
            }
        } else if (type === "status") {
            updateBotInfo(data);
        }

        addEvent(type, JSON.stringify(data).substring(0, 200));
    }

    function addEvent(type, msg) {
        const entry = document.createElement("div");
        entry.className = "log-entry";
        const now = new Date().toLocaleTimeString();
        entry.innerHTML = `<span class="time">${now}</span> <span class="type">[${type}]</span> <span class="msg">${escapeHtml(msg)}</span>`;
        eventLog.appendChild(entry);
        eventLog.scrollTop = eventLog.scrollHeight;

        // Keep max 100 entries
        while (eventLog.children.length > 100) {
            eventLog.removeChild(eventLog.firstChild);
        }
    }

    function addChatMessage(sender, text, cls) {
        const msg = document.createElement("div");
        msg.className = `chat-msg ${cls}`;
        msg.innerHTML = `<div class="sender">${escapeHtml(sender)}</div><div>${escapeHtml(text)}</div>`;
        chatLog.appendChild(msg);
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    function updateBotInfo(data) {
        const el = (id) => document.getElementById(id);
        if (data.status) el("bot-status").textContent = data.status;
        if (data.model) el("bot-model").textContent = data.model;
        if (data.uptime) el("bot-uptime").textContent = data.uptime;
        if (data.memories !== undefined) el("bot-memories").textContent = data.memories;
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    // Chat send
    function sendChat() {
        const text = chatInput.value.trim();
        if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

        ws.send(JSON.stringify({ type: "chat", text: text }));
        addChatMessage("You", text, "user");
        chatInput.value = "";
    }

    chatSend.addEventListener("click", sendChat);
    chatInput.addEventListener("keypress", function(e) {
        if (e.key === "Enter") sendChat();
    });

    // Initial REST fetch
    async function fetchHealth() {
        try {
            const r = await fetch("/health");
            const data = await r.json();
            document.getElementById("bot-status").textContent = data.status || "running";
        } catch(e) {
            document.getElementById("bot-status").textContent = "API error";
        }
    }

    // Start
    fetchHealth();
    connectWS();
})();
