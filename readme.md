# Endless Reflector

An art piece powered by a small local LLM.  
It thinks out loud in short reflections, streaming them to a web page that looks like an old terminal.  
Everyone who opens the page can witness its live output.

---

## Features

- Runs a local [Ollama](https://ollama.ai/) model (default: `llama3.2:3b`).
- Endless self-reflection loop, generating short, varied paragraphs.
- Output streams live to all connected browsers.
- Terminal-style web UI with uptime, idle/thinking status, and blinking cursor.
- Thoughts reset whenever the app restarts.
- Optional `/reset` API endpoint to clear memory on demand.

---

## Requirements

- Docker and Docker Compose (new style, without `version:` line).
- At least a quad-core CPU and 4 GB RAM (the model runs tight).

---

## Quick Start

Clone or copy this project and start the services:

    docker compose up -d --build

This will start two containers:
- `ollama` → the model runtime.
- `reflector` → the FastAPI server and web frontend.

By default, the web app listens on **http://localhost:8080**.

---

## Configuration

You can tweak behavior using environment variables in `docker-compose.yml`:

| Variable                  | Default            | Description |
|---------------------------|--------------------|-------------|
| `MODEL`                   | `llama3.2:3b`      | Ollama model to run. |
| `OLLAMA_URL`              | `http://ollama:11434` | Internal URL of Ollama. |
| `REFLECTION_INTERVAL_SECONDS` | `30`         | Delay between new reflections. |
| `MAX_CONTEXT_TOKENS`      | `6000`             | Max tokens kept as memory. |
| `SUMMARIZE_EVERY_N_TURNS` | `20`               | Summarize every N entries (internal). |

---

## Resetting

The model has no permanent memory, but you can force a reset (clear DB + Ollama session):

### Using curl

    curl -X POST http://localhost:8080/reset

### From the browser console

    fetch("/reset", {method: "POST"});

After reset, the web UI will clear and fresh reflections will begin.

---

## Development

To watch logs live:

    docker compose logs -f

To rebuild after code changes:

    docker compose up -d --build reflector

---

## Notes

- This is not meant for production load. It’s intentionally fragile and poetic.
- Output is nondeterministic and sometimes repetitive. The system prompt biases it toward short, varied reflections.
- ASCII art may appear occasionally if the model chooses.
- Closing all browsers doesn’t stop it; the loop continues until reset.

---

Enjoy the show.
