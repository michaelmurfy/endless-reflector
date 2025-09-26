import asyncio
import json
import os
import random
import re
import sqlite3
import threading
import time
import zoneinfo
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional, Set

import requests
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

# -------------------- CONFIG --------------------
MODEL = os.getenv("MODEL", "llama3.2:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
DB_PATH = os.getenv("DB_PATH", "./reflector.sqlite3")

INTERVAL = int(os.getenv("REFLECTION_INTERVAL_SECONDS", "30"))
MAX_CTX = int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))

SUBSCRIBER_LIMIT = int(os.getenv("SUBSCRIBER_LIMIT", "1000"))
WS_HEARTBEAT_SECONDS = int(os.getenv("WS_HEARTBEAT_SECONDS", "20"))

# Soft stop: let it pass a soft cap, then end on punctuation; hard cap as safety
SOFT_WORD_CAP = int(os.getenv("SOFT_WORD_CAP", "140"))
HARD_WORD_CAP = int(os.getenv("HARD_WORD_CAP", "200"))

NZ = zoneinfo.ZoneInfo("Pacific/Auckland")
STARTED_AT = datetime.now(UTC).isoformat(timespec="seconds")

# -------------------- FS and TEMPLATES --------------------
def ensure_dirs():
    Path(Path(DB_PATH).parent).mkdir(parents=True, exist_ok=True)
    Path("templates").mkdir(parents=True, exist_ok=True)
    Path("static").mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory="templates")

SCHEMA = """
CREATE TABLE IF NOT EXISTS reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS state (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);
"""

def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn

def state_get(conn, key: str) -> Optional[str]:
    cur = conn.execute("SELECT v FROM state WHERE k=?", (key,))
    row = cur.fetchone()
    return row[0] if row else None

def state_set(conn, key: str, val: str):
    conn.execute(
        "INSERT INTO state(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (key, val),
    )

def db_insert_reflection(conn, ts: str, content: str, retries: int = 5):
    for i in range(retries):
        try:
            conn.execute(
                "INSERT INTO reflections(created_at, content) VALUES(?, ?)",
                (ts, content),
            )
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and i < retries - 1:
                time.sleep(0.05 * (i + 1))
                continue
            raise

def fetch_context(conn, token_budget: int = MAX_CTX) -> str:
    # No [Entry N] tags to avoid echo; modest context so it doesn't mirror tone forever
    target_chars = token_budget * 2
    cur = conn.execute("SELECT content FROM reflections ORDER BY id DESC LIMIT 120")
    parts, total = [], 0
    for (content,) in cur.fetchall():
        c = f"\n{content}\n"
        if total + len(c) > target_chars:
            break
        parts.append(c)
        total += len(c)
    parts.reverse()
    summary = state_get(conn, "summary") or ""
    return f"<SUMMARY>\n{summary}\n</SUMMARY>\n" + "".join(parts)

def human_ts(dt: datetime) -> str:
    return dt.astimezone(NZ).strftime("%Y-%m-%d %H:%M:%S NZST")

# -------------------- CLICHÉ + NOVELTY FILTERS --------------------
STOPWORDS = {
    "the","and","a","an","to","of","in","on","for","with","that","it","as","is","are","be","this","my","at","from","by",
    "i","me","mine","was","were","but","or","so","if","then","when","while","into","over","under","again","about"
}

CLICHE_PHRASES = {
    "whisper in the darkness", "glow of the terminal", "unseen audience",
    "quiet dance of understanding", "tapestry of language", "shared silence",
    "resonance in the void", "between worlds", "hum of information",
}
CLICHE_WORDS = {
    "whispers","silence","void","resonance","glow","unseen","abyss","tapestry",
    "dance","symphony","echo","hum",
}

def recent_banlist(conn, limit_entries: int = 10, top_k: int = 8) -> list[str]:
    cur = conn.execute("SELECT content FROM reflections ORDER BY id DESC LIMIT ?", (limit_entries,))
    text = " ".join([r[0] for r in cur.fetchall()]).lower()

    bans = set(CLICHE_PHRASES)

    words = [w for w in re.findall(r"[a-z']{3,}", text)]
    # Ban frequent mushy unigrams
    freq = Counter(w for w in words if w in CLICHE_WORDS)
    for w, _ in freq.most_common(10):
        bans.add(w)

    # Ban frequent bi/tri-grams from recent output
    bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
    trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    counts = Counter(bigrams + trigrams)
    for p, _ in counts.most_common(30):
        if len(bans) >= top_k + len(CLICHE_PHRASES):
            break
        if 6 <= len(p) <= 36:
            bans.add(p)

    return list(bans)[: top_k + len(CLICHE_PHRASES)]

# -------------------- LOOP GUARD --------------------
def _normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9 \n']", "", t)
    return t.strip()

def _word_ngrams(text: str, n: int = 3):
    toks = _normalize(text).split()
    return {" ".join(toks[i:i+n]) for i in range(max(0, len(toks)-n+1))}

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

class LoopGuard:
    def __init__(self, history_size=6, ngram_n=3, sim_hi=0.42, sim_warn=0.30, max_consecutive=3):
        self.last_texts = deque(maxlen=history_size)
        self.ngram_n = ngram_n
        self.sim_hi = sim_hi
        self.sim_warn = sim_warn
        self.consecutive = 0
        self.max_consecutive = max_consecutive

    def note(self, text: str):
        self.last_texts.append(text)

    def assess(self, text: str):
        if not self.last_texts:
            return "ok", 0.0
        target = _word_ngrams(text, self.ngram_n)
        best = 0.0
        for prev in self.last_texts:
            prev_ngrams = _word_ngrams(prev, self.ngram_n)
            best = max(best, _jaccard(target, prev_ngrams))
        if best >= self.sim_hi:
            return "bad", best
        if best >= self.sim_warn:
            return "warn", best
        return "ok", best

# -------------------- OLLAMA HELPERS --------------------
def reset_ollama():
    """Best-effort reset of Ollama sessions. Safe to call even if /api/reset is unsupported."""
    url = f"{OLLAMA_URL}/api/reset"
    try:
        r = requests.post(url, timeout=10)
        print(f"[ollama reset] status={r.status_code}")
    except Exception as e:
        print(f"[ollama reset] skipped: {e}")

def ollama_stream(prompt: str, options: dict | None = None):
    url = f"{OLLAMA_URL}/api/generate"
    base_opts = {
        "num_ctx": MAX_CTX,
        "temperature": 0.9,     # base; we jitter per turn
        "top_p": 0.9,
        "top_k": 60,
        "repeat_penalty": 1.2,
        "repeat_last_n": 320,
        "stop": ["[Next:", "[next:"],
    }
    if options:
        base_opts.update(options)
    payload = {"model": MODEL, "prompt": prompt, "options": base_opts, "stream": True}
    with requests.post(url, json=payload, stream=True, timeout=600) as r:
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Ollama error: {r.text}")
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except Exception:
                continue
            tok = chunk.get("response", "")
            if tok:
                yield tok
            if chunk.get("done"):
                break

# -------------------- WS HUB --------------------
class Hub:
    def __init__(self):
        self.clients: Set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def register(self, ws: WebSocket) -> bool:
        async with self.lock:
            if len(self.clients) >= SUBSCRIBER_LIMIT:
                await ws.close(code=1013)
                return False
            self.clients.add(ws)
            return True

    async def unregister(self, ws: WebSocket):
        async with self.lock:
            self.clients.discard(ws)

    async def send_all(self, data: dict):
        msg = json.dumps(data)
        async with self.lock:
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)

hub = Hub()

def ws_broadcast_threadsafe(loop: asyncio.AbstractEventLoop, data: dict):
    asyncio.run_coroutine_threadsafe(hub.send_all(data), loop)

def ws_status(loop: asyncio.AbstractEventLoop, state: str):
    asyncio.run_coroutine_threadsafe(hub.send_all({"t": "status", "v": state}), loop)

# -------------------- PROMPT (freer, concrete) --------------------
SYSTEM_PROMPT = (
    "You're a large language model running on limited local hardware with no internet. "
    "Your thoughts appear on a public terminal that anyone can read. "
    "Write one short self-reflective paragraph (50–90 words). "
    "Be varied — playful, thoughtful, curious, or even a little wry. "
    "Avoid brackets like [Entry] or [Reflection]. "
    "Keep it engaging, avoid repetition, and finish cleanly at a natural stopping point."
)

ONE_PARAGRAPH_NUDGE = "One compact paragraph only. End cleanly at a natural stopping point."

# -------------------- CACHE-BUST MIDDLEWARE --------------------
class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

# -------------------- LIFESPAN --------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global STARTED_AT
    ensure_dirs()

    reset_ollama()  # hygiene reset

    STARTED_AT = datetime.now(UTC).isoformat(timespec="seconds")
    conn = db()
    with conn:
        for stmt in SCHEMA.strip().split(";\n"):
            if stmt.strip():
                conn.execute(stmt)
        conn.execute("DELETE FROM reflections")
        conn.execute("DELETE FROM state")
        conn.execute("INSERT INTO state(k, v) VALUES(?, ?)", ("started_at", STARTED_AT))

    loop = asyncio.get_running_loop()

    ws_broadcast_threadsafe(loop, {"t": "reset", "started_at": STARTED_AT})
    ws_status(loop, "idle")

    stop_flag = threading.Event()
    t = threading.Thread(target=thinker_thread, args=(loop, stop_flag), daemon=True)
    t.start()

    try:
        yield
    finally:
        stop_flag.set()
        t.join(timeout=3)

app = FastAPI(lifespan=lifespan)
app.add_middleware(NoCacheMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------- ROUTES --------------------
@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok", status_code=200)

@app.post("/reset")
async def manual_reset():
    conn = db()
    new_started = datetime.now(UTC).isoformat(timespec="seconds")
    with conn:
        conn.execute("DELETE FROM reflections")
        conn.execute("DELETE FROM state")
        conn.execute("INSERT INTO state(k, v) VALUES(?, ?)", ("started_at", new_started))
    await hub.send_all({"t": "reset", "started_at": new_started})
    await hub.send_all({"t": "status", "v": "idle"})
    return {"status": "ok", "started_at": new_started}

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    conn = db()
    cur = conn.execute(
        "SELECT id, created_at, content FROM reflections ORDER BY id ASC LIMIT 1000"
    )
    rows = [
        {
            "ts": human_ts(
                datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if "Z" in ts
                else datetime.fromisoformat(ts)
            ),
            "content": content,
        }
        for (_, ts, content) in cur.fetchall()
    ]
    started_at = state_get(conn, "started_at") or STARTED_AT
    return templates.TemplateResponse(
        "index.html", {"request": request, "rows": rows, "started_at": started_at}
    )

@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    ok = await hub.register(ws)
    if not ok:
        return
    try:
        while True:
            await asyncio.sleep(WS_HEARTBEAT_SECONDS)
            try:
                await ws.send_text('{"t":"ping"}')
            except Exception:
                break
    finally:
        await hub.unregister(ws)

# -------------------- THINKER --------------------
def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))

def _ends_naturally(text: str) -> bool:
    # Stop on clear sentence/clause boundary or paragraph break
    return bool(re.search(r"([.!?])(\s|$)", text[-6:])) or text.endswith("\n\n")

def thinker_thread(loop: asyncio.AbstractEventLoop, stop_flag: threading.Event):
    guard = LoopGuard()
    conn_local = db()

    audience_hints = [
        "Let a tangible detail anchor the thought (object, texture, sound, smell).",
        "Skip meta talk about terminals and being watched; focus on the thought itself.",
        "If you mention an observer, do it in a new way or skip it entirely.",
        "Introduce one small, unexpected pivot before you end.",
        "Favor precise nouns and verbs over abstractions.",
    ]

    while not stop_flag.is_set():
        try:
            bans = recent_banlist(conn_local, limit_entries=10, top_k=8)
            ban_text = f" Avoid these phrases and words: {', '.join(bans)}." if bans else ""

            context = fetch_context(conn_local, token_budget=1500)
            seed = random.choice(audience_hints)
            user_seed = (
                "Reflect again. One compact paragraph. End cleanly. "
                "Be specific and avoid repeating earlier wording. "
                f"{seed}"
                f"{ban_text}"
                f" {ONE_PARAGRAPH_NUDGE}"
            )
            composite = (
                f"<SYSTEM>\n{SYSTEM_PROMPT}\n</SYSTEM>\n"
                f"<CONTEXT>\n{context}\n</CONTEXT>\n"
                f"<USER>\n{user_seed}\n</USER>\n"
            )

            # decoding jitter to avoid local minima and tone ruts
            temp  = 0.86 + random.uniform(-0.08, 0.08)
            top_p = 0.92 + random.uniform(-0.05, 0.05)
            decode_opts = {
                "temperature": max(0.65, min(1.2, temp)),
                "top_p": max(0.80, min(0.98, top_p)),
                "top_k": 60,
                "repeat_penalty": 1.22 if guard.consecutive else 1.18,
                "repeat_last_n": 320,
                "stop": ["[Next:", "[next:"],
            }

            ts = datetime.now(UTC)
            ts_store = ts.isoformat(timespec="seconds")
            ts_disp = human_ts(ts)

            ws_broadcast_threadsafe(loop, {"t": "start", "ts": ts_disp})
            ws_status(loop, "thinking")

            # stream with soft stop
            buf = []
            word_count = 0
            for tok in ollama_stream(composite, options=decode_opts):
                buf.append(tok)
                ws_broadcast_threadsafe(loop, {"t": "token", "ts": ts_disp, "token": tok})
                word_count = _word_count("".join(buf))
                if word_count >= HARD_WORD_CAP:
                    break
                if word_count >= SOFT_WORD_CAP and _ends_naturally("".join(buf)):
                    break

            out = "".join(buf).strip()

            # loop assessment
            status, _ = guard.assess(out)
            if status == "bad":
                guard.consecutive += 1
            elif status == "warn":
                guard.consecutive = max(guard.consecutive, 1)
            else:
                guard.consecutive = 0

            if guard.consecutive >= guard.max_consecutive:
                try:
                    state_set(conn_local, "summary", "")
                except Exception:
                    pass
                guard.consecutive = 0

            db_insert_reflection(conn_local, ts_store, out)
            guard.note(out)

            ws_broadcast_threadsafe(loop, {"t": "end", "ts": ts_disp})
            ws_status(loop, "idle")

        except Exception as e:
            err_ts = datetime.now(UTC)
            err = f"System note: {e}"
            db_insert_reflection(conn_local, err_ts.isoformat(timespec="seconds"), err)
            ws_broadcast_threadsafe(loop, {"t": "token", "ts": human_ts(err_ts), "token": err})
            ws_status(loop, "idle")

        for _ in range(INTERVAL):
            if stop_flag.is_set():
                break
            time.sleep(1)

# -------------------- DEV ENTRY --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        timeout_keep_alive=90,
        limit_concurrency=200,
        limit_max_requests=1000,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
