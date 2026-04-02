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
            showError("error-msg", "ЭТОТ ЭКИПАЖ УЖЕ НА БОРТУ");
        } else if (err.message.includes("404")) {
            showError("error-msg", "ПОЗЫВНОЙ НЕ НАЙДЕН");
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
function initSessionPage() {
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
}

async function sendHeartbeat() {
    if (!currentSession) return;
    try {
        await api("POST", "/api/sessions/heartbeat", { code: currentSession.code });
    } catch (err) { }
}

const SECTION_TITLES = [
    "СУТЬ ПОЗИЦИИ",
    "КЛЮЧЕВЫЕ АРГУМЕНТЫ",
    "ЧТО ПРОИЗОЙДЁТ К 2100",
    "ГЛАВНЫЕ РИСКИ",
    "ЧТО СКАЗАТЬ НА ДЕБАТАХ",
    "ВОПРОСЫ ОППОНЕНТАМ",
    "НОВОСТЬ ИЗ 2100 ГОДА",
];

const SECTION_KEYS = [
    "position", "arguments", "predictions", "risks",
    "debate_speech", "opponent_questions", "news_2100",
];

function renderAnswer(answer) {
    let html = "";
    for (let i = 0; i < SECTION_KEYS.length; i++) {
        const text = answer[SECTION_KEYS[i]];
        if (!text) continue;
        html += `<div class="section-block">
            <div class="section-title">${SECTION_TITLES[i]}</div>
            <div class="section-content">${escapeHtml(text)}</div>
        </div>`;
    }
    return html;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
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
    document.getElementById("loading-indicator").style.display = "block";

    const chatArea = document.getElementById("chat-area");

    try {
        const data = await api("POST", "/api/ask", {
            code: currentSession.code,
            question: question,
        });

        const block = document.createElement("div");
        block.className = "qa-block";
        block.innerHTML = `
            <div class="qa-question">
                <span class="prompt-symbol">&gt;</span> ${escapeHtml(question)}
            </div>
            <div class="qa-answer">
                ${renderAnswer(data.answer)}
            </div>
        `;
        chatArea.appendChild(block);
        block.scrollIntoView({ behavior: "smooth" });
    } catch (err) {
        const errBlock = document.createElement("div");
        errBlock.className = "qa-block";
        errBlock.innerHTML = `
            <div class="qa-question">
                <span class="prompt-symbol">&gt;</span> ${escapeHtml(question)}
            </div>
            <div class="qa-answer" style="color: var(--accent);">
                ОШИБКА СВЯЗИ: ${escapeHtml(err.message)}
            </div>
        `;
        chatArea.appendChild(errBlock);
    } finally {
        input.disabled = false;
        document.getElementById("send-btn").disabled = false;
        document.getElementById("loading-indicator").style.display = "none";
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
