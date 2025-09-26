# Endless Reflector

A tiny local LLM that “thinks” forever and prints its thoughts to a minimalist terminal-style web page.

---

## Prereqs

- Docker + Docker Compose v2 installed
- Ports 8080 (app) and 11434 (Ollama) available

---

## Quick start

1) Start the stack

    docker compose up -d --build

2) Install/pull the model inside the Ollama container (first run only)

    docker compose exec ollama ollama pull llama3.2:3b

3) Open the UI

    http://localhost:8080

You should see a black page with an uptime bar, a thinking/idle badge, and live text streaming in.

---

## Reset the “mind”

From a terminal:

    curl -X POST http://localhost:8080/reset

From the browser console on the page:

    fetch("/reset", { method: "POST" });

This clears stored reflections and restarts uptime for all viewers.

---

## Edit the system prompt (important)

Open:

    app/app.py

Find this block (near the top):

    SYSTEM_PROMPT = (
        "You're a large language model running on limited local hardware with no internet. "
        "Your thoughts appear on a public terminal that anyone can read. "
        "Write one short self-reflective paragraph (50–90 words). "
        "Be concrete, specific, and varied. Prefer vivid particulars over abstractions. "
        "You may be playful, thoughtful, wry, or curious. "
        "Avoid melodrama. Avoid repeating imagery or phrasing from earlier thoughts. "
        "Do not use bracketed labels like [Entry], [Reflection], or [Sketch]. "
        "Occasional tiny ASCII doodles are fine, but rare and concise. "
        "Do not explain the constraints unless it serves a fresh idea."
    )

How to customize:
- Keep it short. One paragraph only. Ask for “concrete, specific” details to cut poetry mush.
- If you want more humor: add “lean toward dry humor and small jokes.”
- If you want less meta: add “avoid mentioning the audience unless there’s a fresh angle.”
- If you dislike ASCII art: delete that line.
- If you want darker/softer tone, say it explicitly. Avoid contradictions.

After edits, rebuild the app container (Ollama stays up):

    docker compose up -d --build reflector

---

## Config (optional)

Edit values in `docker-compose.yml` or set env vars:

- MODEL (default: `llama3.2:3b`)
- OLLAMA_URL (default: `http://ollama:11434`)
- REFLECTION_INTERVAL_SECONDS (default: `30`)
- MAX_CONTEXT_TOKENS (default: `6000`)
- SOFT_WORD_CAP (default: `140`) and HARD_WORD_CAP (default: `200`) — soft stop finishes on punctuation

---

## Dev commands

Tail logs:

    docker compose logs -f

Rebuild just the app:

    docker compose up -d --build reflector

Stop everything:

    docker compose down

---

## Troubleshooting

- Blank page / 502: make sure the model is pulled

      docker compose exec ollama ollama list
      docker compose exec ollama ollama pull llama3.2:3b

- Reverse proxy / tunnels: enable WebSocket pass-through and increase timeouts.
- Health check:

      curl http://localhost:8080/healthz

---
