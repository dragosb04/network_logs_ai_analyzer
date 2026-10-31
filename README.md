# Log Summary + AI Reviewer

A local tool that analyzes log files and asks a locally hosted LLM what the likely problem is.
It counts ERROR / WARNING / INFO entries, groups the most frequent errors and shows an AI suggestion
(main issue, confidence, possible causes, what to check next) in a simple web interface.

Everything runs on your machine: the log is never sent to an external service.

> The AI generated output is a suggestion only. A human must verify it.

## Features

- Log statistics and the most frequent error messages
- AI analysis with a fixed structure (Pydantic structured output via LangChain)
- **LangGraph agent loop**: if the model reports low confidence, the agent adds log lines around the errors and retries
- Masks IP addresses, emails and credentials before anything reaches the model
- Drag-and-drop web interface
- Terminal logs for every step (sizes and decisions only, never log content)

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

## Installation

```bash
git clone https://github.com/dragosb04/network_logs_ai_analyzer.git
cd network_logs_ai_analyzer

pip install langgraph langchain-ollama

ollama pull qwen2.5:3b
```

## Usage

1. Make sure Ollama is running:
   ```bash
   ollama serve
   ```
2. Start the app:
   ```bash
   python log_ai_reviewer.py
   ```
3. The browser opens at `http://127.0.0.1:8765`.
4. Drag a `.log` file into the page (or click to choose one) and press **Analizează**.

### Using a different model

The default model is `qwen2.5:3b`. To use another one, pull it and set `OLLAMA_MODEL`:

```bash
ollama pull deepseek-r1:1.5b
OLLAMA_MODEL=deepseek-r1:1.5b python log_ai_reviewer.py
```

On Windows (PowerShell):

```powershell
$env:OLLAMA_MODEL="deepseek-r1:1.5b"; python log_ai_reviewer.py
```

Smaller models are faster but follow the structured format less reliably.

## How it works

```
log file
   │
   ▼
summarize            count levels, group errors
   │
   ▼
build_input   ◄──────────────┐   summary (+ log lines around errors on retry), masked
   │                         │
   ▼                         │
ask_ai                       │   LangChain + Ollama, structured output
   │                         │
   ▼                         │
decide ── Low confidence ──► more_context  (±3 more lines, max 2 attempts)
   │
   ▼ otherwise
result shown in the browser
```

The agent is built with LangGraph: each step is a node, and `decide` is a conditional edge
that either finishes or loops back with more context.

Settings at the top of `log_ai_reviewer.py`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `MAX_AI_ATTEMPTS` | 2 | first answer + retries |
| `CONTEXT_STEP` | 3 | lines added before/after each ERROR on a retry |
| `MAX_CONTEXT_CHARS` | 3000 | limit for raw log text in the prompt |

## Privacy

- IP addresses → `<IP>`, emails → `<EMAIL>`, `password` / `token` / `secret` / `api_key` values → `<HIDDEN>`
- Masking happens right before the prompt is built, including on retries
- The server listens only on `127.0.0.1`
- Terminal logs contain counts, sizes and decisions, not the log content

## Troubleshooting

| Message | Fix |
| --- | --- |
| Nu mă pot conecta la Ollama | Start Ollama with `ollama serve` |
| Ollama a returnat o eroare | The model is not downloaded: `ollama pull <model>` |
| Modelul nu a returnat un răspuns JSON valid | Try again or use a larger model (e.g. `qwen2.5:3b` or `qwen2.5:7b`) |

## Test data

No logs of your own? `generate_data.py` creates a sample log you can use to try the tool:

```bash
python3 generate_data.py
```

On Windows, use `python` instead of `python3`.

It writes `datasets.log` (2,000 lines) in the current folder. Running it again overwrites the file.

The generated log contains:
- `INFO`, `WARNING` and `ERROR` entries in the supported format, a few seconds apart
- common problems such as refused connections, DNS failures, a full disk, timeouts and an exhausted database pool
- fake IP addresses, emails and passwords, so you can check that masking works before anything reaches the model

Entries are picked at random, so every run produces a slightly different log. All data is fictional.

## Supported log format

The tool currently expects one entry per line with an uppercase level keyword:

    2026-10-07 10:15:36 ERROR Cannot write to /var/data/cache: No space left on device

- Recognized levels: `ERROR`, `WARNING`, `INFO`
- The error message is the text after `ERROR`

### Known limitations

- Lowercase or short levels (`error`, `WARN`, `ERR`, `FATAL`) are not detected
- JSON logs (`{"level": "error", ...}`) and other formats are not parsed
- Messages that differ only by IP or PID are counted as different errors

Planned: a configurable parser for more log formats.
