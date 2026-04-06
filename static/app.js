/* ============================================================
   ФОРСАЙТ-СТАНЦИЯ — Client Logic
   ============================================================ */

let currentSession = null;
let heartbeatInterval = null;

async function api(method, path, body = null) {
    const opts = {
        method,
        headers: { "Content-Type": "application/json" },
    };
    if (body) opts.body = JSON.stringify(body);
    const resp = await fetch(path, opts);
    const data = await resp.json();
    if (!resp.ok) {
        throw new Error(data.detail || `HTTP ${resp.status}`);
    }
    return data;
}

function showError(elementId, msg) {
    const el = document.getElementById(elementId);
    if (el) {
        el.textContent = msg;
        el.style.display = "block";
        setTimeout(() => { el.style.display = "none"; }, 5000);
    }
}

// --- Entry Page ---
async function joinSession() {
    const codeInput = document.getElementById("session-code");
    if (!codeInput) return;
    const code = codeInput.value.trim().toUpperCase();
    if (!code) return;

    try {
        const data = await api("POST", "/api/sessions/join", { code });
        sessionStorage.setItem("session", JSON.stringify(data));
        window.location.href = "/session";
    } catch (err) {
        if (err.message.includes("409")) {
            showError("error-msg", "УЖЕ НА БОРТУ");
        } else if (err.message.includes("404")) {
            showError("error-msg", "НЕ НАЙДЕН");
        } else {
            showError("error-msg", err.message);
        }
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const codeInput = document.getElementById("session-code");
    if (codeInput) {
        codeInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") joinSession();
        });
    }

    const chatArea = document.getElementById("chat-area");
    if (chatArea) {
        initSessionPage();
    }

    const qInput = document.getElementById("question-input");
    if (qInput) {
        qInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") askQuestion();
        });
    }

    const pinInput = document.getElementById("pin-input");
    if (pinInput) {
        pinInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") adminLogin();
        });
    }
});

// --- Session Page ---
async function initSessionPage() {
    const raw = sessionStorage.getItem("session");
    if (!raw) {
        window.location.href = "/";
        return;
    }
    currentSession = JSON.parse(raw);

    document.getElementById("badge-code").textContent = currentSession.code;
    document.getElementById("mission-topic").textContent = currentSession.topic;

    const roleNames = { optimist: "ТЕХНООПТИМИСТ", pessimist: "ТЕХНОПЕССИМИСТ" };
    document.getElementById("role-name").textContent = roleNames[currentSession.role];

    document.body.classList.add(`role-${currentSession.role}`);

    heartbeatInterval = setInterval(sendHeartbeat, 30000);
    sendHeartbeat();

    await loadChatHistory();
}

async function loadChatHistory() {
    try {
        const data = await api("GET", `/api/sessions/${currentSession.code}/history`);
        if (data.questions.length === 0) return;

        const welcome = document.getElementById("welcome-msg");
        if (welcome) welcome.remove();

        const chatArea = document.getElementById("chat-area");
        for (const q of data.questions) {
            const userBubble = document.createElement("div");
            userBubble.className = "chat-msg chat-msg-user";
            userBubble.innerHTML = `<div class="bubble-user">${escapeHtml(q.question)}</div>`;
            chatArea.appendChild(userBubble);

            const aiBubble = document.createElement("div");
            aiBubble.className = "chat-msg chat-msg-ai";
            aiBubble.innerHTML = `<div class="bubble-ai">${renderAnswer(q.answer)}</div>`;
            chatArea.appendChild(aiBubble);
        }

        // Scroll to bottom without animation (restoring state)
        chatArea.scrollTop = chatArea.scrollHeight;
    } catch (err) { }
}

async function sendHeartbeat() {
    if (!currentSession) return;
    try {
        await api("POST", "/api/sessions/heartbeat", { code: currentSession.code });
    } catch (err) { }
}

function renderAnswer(answer) {
    // Combine all non-empty fields into one flowing text
    const parts = [
        answer.position, answer.arguments, answer.predictions,
        answer.risks, answer.debate_speech, answer.opponent_questions,
        answer.news_2100,
    ].filter(Boolean);
    return `<div class="answer-text">${renderMarkdown(parts.join("\n\n"))}</div>`;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function renderMarkdown(text) {
    let html = escapeHtml(text);
    // Bold: **text** or __text__
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');
    // Italic: *text* or _text_
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    // Inline code
    html = html.replace(/`(.+?)`/g, '<code>$1</code>');
    // Headers (strip ##, make bold)
    html = html.replace(/^#{1,3}\s+(.+)$/gm, '<strong>$1</strong>');
    // Numbered lists: 1. item
    html = html.replace(/^\d+\.\s+(.+)$/gm, '<div class="md-li">$&</div>');
    // Bullet lists: - item or * item
    html = html.replace(/^[\-\*]\s+(.+)$/gm, '<div class="md-li">&bull; $1</div>');
    // Quotes: > text
    html = html.replace(/^&gt;\s+(.+)$/gm, '<div class="md-quote">$1</div>');
    // Paragraphs: split on double newlines
    html = html.split(/\n{2,}/).map(p => {
        p = p.trim();
        if (!p) return "";
        if (p.startsWith("<div")) return p;  // already a block element
        return `<div class="md-p">${p.replace(/\n/g, '<br>')}</div>`;
    }).join("");
    return html;
}

async function askQuestion() {
    if (!currentSession) return;
    const input = document.getElementById("question-input");
    const question = input.value.trim();
    if (!question) return;

    const welcome = document.getElementById("welcome-msg");
    if (welcome) welcome.remove();

    input.value = "";
    input.disabled = true;
    document.getElementById("send-btn").disabled = true;

    const chatArea = document.getElementById("chat-area");

    // User bubble — appears immediately on the right
    const userBubble = document.createElement("div");
    userBubble.className = "chat-msg chat-msg-user";
    userBubble.innerHTML = `<div class="bubble-user">${escapeHtml(question)}</div>`;
    chatArea.appendChild(userBubble);

    // Thinking indicator — left side, while waiting
    const thinkingBubble = document.createElement("div");
    thinkingBubble.className = "chat-msg chat-msg-ai";
    thinkingBubble.innerHTML = `<div class="bubble-thinking"><span class="blink">БОРТОВОЙ КОМПЬЮТЕР ОБРАБАТЫВАЕТ ЗАПРОС...</span></div>`;
    chatArea.appendChild(thinkingBubble);
    thinkingBubble.scrollIntoView({ behavior: "smooth" });

    try {
        const data = await api("POST", "/api/ask", {
            code: currentSession.code,
            question: question,
        });

        thinkingBubble.remove();

        const aiBubble = document.createElement("div");
        aiBubble.className = "chat-msg chat-msg-ai";
        aiBubble.innerHTML = `<div class="bubble-ai">${renderAnswer(data.answer)}</div>`;
        chatArea.appendChild(aiBubble);
        aiBubble.scrollIntoView({ behavior: "smooth" });
    } catch (err) {
        thinkingBubble.remove();

        const errBubble = document.createElement("div");
        errBubble.className = "chat-msg chat-msg-ai";
        errBubble.innerHTML = `<div class="bubble-ai" style="color: var(--accent);">ОШИБКА СВЯЗИ: ${escapeHtml(err.message)}</div>`;
        chatArea.appendChild(errBubble);
        errBubble.scrollIntoView({ behavior: "smooth" });
    } finally {
        input.disabled = false;
        document.getElementById("send-btn").disabled = false;
        input.focus();
    }
}

// --- Admin Page ---
async function adminLogin() {
    const pin = document.getElementById("pin-input").value;
    try {
        await api("POST", "/api/admin/login", { pin });
        document.getElementById("pin-overlay").style.display = "none";
        document.getElementById("dashboard").style.display = "block";
        refreshDashboard();
        // Auto-refresh every 5 seconds
        setInterval(refreshDashboard, 5000);
    } catch (err) {
        showError("pin-error", "НЕВЕРНЫЙ КОД ДОСТУПА");
    }
}

async function createAllSessions() {
    try {
        await api("POST", "/api/sessions/create-all");
        refreshDashboard();
    } catch (err) {
        alert("Ошибка: " + err.message);
    }
}

async function resetAllSessions() {
    if (!confirm("ВНИМАНИЕ: Все активные сессии будут сброшены. Продолжить?")) return;
    try {
        await api("POST", "/api/sessions/reset-all");
        refreshDashboard();
    } catch (err) {
        alert("Ошибка: " + err.message);
    }
}

async function refreshDashboard() {
    try {
        const sessions = await api("GET", "/api/sessions");
        renderSessionsGrid(sessions);
    } catch (err) { }
}

function renderSessionsGrid(sessions) {
    const grid = document.getElementById("sessions-grid");
    if (!grid) return;

    grid.innerHTML = sessions.map(s => {
        const roleClass = s.role === "optimist" ? "card-optimist" : "card-pessimist";
        const roleLabel = s.role === "optimist" ? "ОПТИМИСТ" : "ПЕССИМИСТ";
        const statusLabel = { waiting: "Ожидает", active: "Онлайн", disconnected: "Отключена" };
        return `<div class="session-card ${roleClass}" onclick="openSessionDetail('${s.code}')">
            <div class="card-code">${s.code}</div>
            <div class="card-topic">${escapeHtml(s.topic)}</div>
            <div class="card-status">
                <span class="status-dot ${s.status}"></span>
                <span>${statusLabel[s.status] || s.status}</span>
                <span style="margin-left:auto">${roleLabel}</span>
            </div>
            <div class="card-questions">Вопросов: ${s.question_count}</div>
        </div>`;
    }).join("");
}

async function openSessionDetail(code) {
    try {
        const data = await api("GET", `/api/sessions/${code}/history`);
        document.getElementById("detail-title").textContent =
            `${data.code} — ${data.topic}`;

        const historyEl = document.getElementById("detail-history");
        if (data.questions.length === 0) {
            historyEl.innerHTML = '<p style="color:var(--text-dim)">Вопросов пока нет.</p>';
        } else {
            historyEl.innerHTML = data.questions.map(q => `
                <div class="detail-qa">
                    <div class="detail-question">&gt; ${escapeHtml(q.question)}</div>
                    <div class="detail-answer">${escapeHtml(q.answer.position || '')}</div>
                </div>
            `).join("");
        }

        document.getElementById("session-detail").style.display = "block";
    } catch (err) {
        alert("Ошибка: " + err.message);
    }
}

function closeDetail() {
    document.getElementById("session-detail").style.display = "none";
}
