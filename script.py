import json
import os
import re
import threading
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from ollama import ResponseError
from pydantic import BaseModel

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:1.5b")
HOST = "127.0.0.1"
PORT = 8765

def summarize(lines):
    level_counts = Counter()
    error_messages = Counter()

    for line in lines:
        if "ERROR" in line:
            level_counts["ERROR"] += 1
            message = line.split("ERROR", 1)[1].strip()
            error_messages[message] += 1
        elif "WARNING" in line:
            level_counts["WARNING"] += 1
        elif "INFO" in line:
            level_counts["INFO"] += 1

    return level_counts, error_messages


def mask_sensitive(text):
    text = re.sub(r"\b\d{1,3}(\.\d{1,3}){3}\b", "<IP>", text)
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "<EMAIL>", text)
    text = re.sub(
        r"(password|passwd|token|secret|api_key)\s*[=:]\s*\S+",
        r"\1=<HIDDEN>",
        text,
        flags=re.IGNORECASE,
    )
    return text


def build_ai_input(level_counts, error_messages):
    parts = [
        f"Errors: {level_counts['ERROR']}, Warnings: {level_counts['WARNING']}",
        "Most frequent errors:",
    ]
    for message, count in error_messages.most_common(10):
        parts.append(f"{count}x {message}")
    return mask_sensitive("\n".join(parts))

class LogAnalysis(BaseModel):
    main_issue: str
    confidence: Literal["High", "Medium", "Low"]
    possible_causes: list[str]
    investigation_steps: list[str]

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert system and network support engineer. Analyze this log summary carefully.\n"
     "Identify the core root cause, evaluate confidence, list likely causes, and actionable checks."),
    ("human", "LOG SUMMARY:\n{summary}"),
])

def build_chain(model_name):
    model = ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        seed=42,
        num_predict=1024,
    )
    structured_model = model.with_structured_output(LogAnalysis)
    return prompt | structured_model

def remove_numbering(text):
    return re.sub(r"^\s*\d+[.)]\s*", "", text)

def ask_ai(summary_text, model_name, max_retries=2):
    chain = build_chain(model_name)
    chain_with_retry = chain.with_retry(
        retry_if_exception_type=(ValueError,),
        stop_after_attempt=max_retries,
    )

    result = chain_with_retry.invoke({"summary": summary_text})
    data = result.model_dump()

    clean_steps = []
    for step in data["investigation_steps"]:
        clean_steps.append(remove_numbering(step))
    data["investigation_steps"] = clean_steps

    return data


def analyze(text, model):
    lines = text.splitlines()
    level_counts, error_messages = summarize(lines)

    result = {
        "total": len(lines),
        "error": level_counts["ERROR"],
        "warning": level_counts["WARNING"],
        "info": level_counts["INFO"],
        "top_errors": [
            {"count": c, "message": m} for m, c in error_messages.most_common(5)
        ],
        "ai": None,
        "ai_error": None,
        "model": model,
    }

    try:
        result["ai"] = ask_ai(build_ai_input(level_counts, error_messages), model)
    except ConnectionError:
        result["ai_error"] = "Nu ma pot conecta la Ollama. Ruleaza? (incearca: ollama serve)"
    except ResponseError:
        result["ai_error"] = f"Ollama a returnat o eroare. Modelul e descarcat? (incearca: ollama pull {model})"
    except ValueError:
        result["ai_error"] = "Modelul nu a returnat un raspuns JSON valid."
    except Exception as error:
        result["ai_error"] = f"Analiza AI a esuat: {error}"

    return result

PAGE = """<!DOCTYPE html>
<html lang="ro">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Log Summary + AI</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --text:#1c2230; --muted:#6b7280; --border:#e2e5ea; --accent:#2563eb; --err:#dc2626; --warn:#d97706; --info:#0891b2; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#12151c; --card:#1b1f29; --text:#e6e8ee; --muted:#9aa1b0; --border:#2c3240; --accent:#60a5fa; --err:#f87171; --warn:#fbbf24; --info:#22d3ee; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui, sans-serif; }
  main { max-width: 860px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 22px; margin: 0 0 16px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:16px; margin-bottom:16px; }
  .card h2 { font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin:0 0 10px; }
  .row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
  input[type=text] { padding:8px 10px; border:1px solid var(--border); border-radius:6px; background:var(--bg); color:var(--text); width:160px; }
  button { padding:9px 18px; border:0; border-radius:6px; background:var(--accent); color:#fff; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  #drop { border:2px dashed var(--border); border-radius:10px; padding:28px; text-align:center; color:var(--muted); cursor:pointer; flex:1 1 100%; }
  #drop.over { border-color:var(--accent); color:var(--accent); }
  .stats { display:flex; gap:12px; flex-wrap:wrap; }
  .stat { flex:1 1 120px; padding:12px; border-radius:8px; background:var(--bg); text-align:center; }
  .stat b { display:block; font-size:24px; }
  .stat span { font-size:12px; color:var(--muted); }
  table { width:100%; border-collapse:collapse; }
  td { padding:8px 6px; border-top:1px solid var(--border); vertical-align:top; word-break:break-word; }
  td.n { width:56px; font-weight:700; text-align:center; }
  .error { color:var(--err); }
  .muted { color:var(--muted); font-size:13px; }
  .hidden { display:none; }

  .ai-box { display: flex; flex-direction: column; gap: 14px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; text-transform: uppercase; }
  .badge-high { background: #fee2e2; color: #b91c1c; }
  .badge-medium { background: #fef3c7; color: #b45309; }
  .badge-low { background: #e0f2fe; color: #0369a1; }
  .ai-section h3 { font-size: 13px; text-transform: uppercase; color: var(--muted); margin: 0 0 8px; }
  .ai-list { margin: 0; padding-left: 20px; }
  .ai-list li { margin-bottom: 6px; }
  .steps-list { list-style: none; padding-left: 0; margin: 0; }
  .steps-list li { padding: 8px 12px; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; margin-bottom: 6px; }
</style>
</head>
<body>
<main>
  <h1>Log Summary + AI Review</h1>

  <div class="card">
    <div class="row">
      <label id="drop" for="file">Trage un fisier .log aici sau click pentru a alege<input id="file" type="file" class="hidden"></label>
      <span id="fname" class="muted">Niciun fisier ales.</span>
    </div>
    <div class="row" style="margin-top:12px">
      <label>Model: <input id="model" type="text" readonly="readonly" value="__MODEL__"></label>
      <button id="go" disabled>Analizeaza</button>
    </div>
  </div>

  <div id="out" class="hidden">
    <div class="card">
      <h2>Statistici</h2>
      <div class="stats">
        <div class="stat"><b id="s-total">-</b><span>Total linii</span></div>
        <div class="stat"><b id="s-error" style="color:var(--err)">-</b><span>ERROR</span></div>
        <div class="stat"><b id="s-warning" style="color:var(--warn)">-</b><span>WARNING</span></div>
        <div class="stat"><b id="s-info" style="color:var(--info)">-</b><span>INFO</span></div>
      </div>
    </div>

    <div class="card">
      <h2>Cele mai frecvente erori</h2>
      <table id="errs"></table>
    </div>

    <!-- [PARTEA 3]: Containerul HTML cu badge si zona de continut -->
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <h2 style="margin:0;">Sugestie AI <span class="muted" id="ai-model"></span></h2>
        <span id="ai-badge" class="badge hidden"></span>
      </div>
      
      <div id="ai-content"></div>
      
      <p class="muted" style="margin-top:14px; margin-bottom:0;">Un om trebuie sa verifice aceasta sugestie.</p>
    </div>
  </div>
</main>

<script>
const $ = id => document.getElementById(id);
let logText = null;

function setFile(file) {
  if (!file) return;
  $("fname").textContent = file.name + " (" + Math.round(file.size / 1024) + " KB)";
  file.text().then(t => { logText = t; $("go").disabled = false; });
}

$("file").addEventListener("change", e => setFile(e.target.files[0]));
["dragover", "dragenter"].forEach(ev => $("drop").addEventListener(ev, e => { e.preventDefault(); $("drop").classList.add("over"); }));
["dragleave", "drop"].forEach(ev => $("drop").addEventListener(ev, e => { e.preventDefault(); $("drop").classList.remove("over"); }));
$("drop").addEventListener("drop", e => setFile(e.dataTransfer.files[0]));

$("go").addEventListener("click", async () => {
  $("go").disabled = true;
  $("out").classList.remove("hidden");
  
  const aiContent = $("ai-content");
  const aiBadge = $("ai-badge");
  aiBadge.className = "badge hidden";
  aiContent.innerHTML = "<em>Se asteapta raspunsul de la model...</em>";
  $("ai-model").textContent = "";

  try {
    const res = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: logText, model: $("model").value.trim() })
    });
    const d = await res.json();
    $("s-total").textContent = d.total;
    $("s-error").textContent = d.error;
    $("s-warning").textContent = d.warning;
    $("s-info").textContent = d.info;

    const t = $("errs");
    t.innerHTML = "";
    if (!d.top_errors.length) {
      t.innerHTML = "<tr><td class='muted'>Nu au fost gasite erori.</td></tr>";
    }
    d.top_errors.forEach(e => {
      const tr = document.createElement("tr");
      const a = document.createElement("td"); a.className = "n"; a.textContent = e.count + "x";
      const b = document.createElement("td"); b.textContent = e.message;
      tr.append(a, b); t.append(tr);
    });

    aiContent.innerHTML = "";
    if (d.ai) {
      $("ai-model").textContent = "(" + d.model + ")";
      const aiData = d.ai;

      if (aiData.confidence) {
        aiBadge.textContent = "Certitudine: " + aiData.confidence;
        aiBadge.classList.remove("hidden");
        const confClass = aiData.confidence.toLowerCase() === "high" ? "badge-high" 
                        : aiData.confidence.toLowerCase() === "low" ? "badge-low" 
                        : "badge-medium";
        aiBadge.classList.add(confClass);
      }

      const box = document.createElement("div");
      box.className = "ai-box";

      if (aiData.main_issue) {
        const issue = document.createElement("div");
        issue.innerHTML = `<strong>Problema identificată:</strong> <p style="margin:4px 0 0;">${aiData.main_issue}</p>`;
        box.appendChild(issue);
      }

      if (aiData.possible_causes && aiData.possible_causes.length) {
        const causesDiv = document.createElement("div");
        causesDiv.className = "ai-section";
        causesDiv.innerHTML = `<h3>Cauze posibile</h3>`;
        const ul = document.createElement("ul");
        ul.className = "ai-list";
        aiData.possible_causes.forEach(cause => {
          const li = document.createElement("li");
          li.textContent = cause;
          ul.appendChild(li);
        });
        causesDiv.appendChild(ul);
        box.appendChild(causesDiv);
      }

      if (aiData.investigation_steps && aiData.investigation_steps.length) {
        const stepsDiv = document.createElement("div");
        stepsDiv.className = "ai-section";
        stepsDiv.innerHTML = `<h3>Ce ar trebui verificat</h3>`;
        const ul = document.createElement("ul");
        ul.className = "steps-list";
        aiData.investigation_steps.forEach((step, idx) => {
          const li = document.createElement("li");
          li.innerHTML = `<strong>${idx + 1}.</strong> ${step}`;
          ul.appendChild(li);
        });
        stepsDiv.appendChild(ul);
        box.appendChild(stepsDiv);
      }

      aiContent.appendChild(box);
    } else {
      aiContent.innerHTML = `<span class="error">${d.ai_error}</span>`;
    }
  } catch (err) {
    aiContent.innerHTML = `<span class="error">Eroare: ${err}</span>`;
  }
  $("go").disabled = false;
});
</script>
</body>
</html>
"""

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE.replace("__MODEL__", DEFAULT_MODEL), "text/html; charset=utf-8")
        else:
            self._send(404, "Not found", "text/plain")

    def do_POST(self):
        if self.path != "/analyze":
            self._send(404, "Not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            text = payload.get("text") or ""
            model = (payload.get("model") or DEFAULT_MODEL).strip()
            result = analyze(text, model)
            self._send(200, json.dumps(result), "application/json")
        except Exception as error:
            self._send(500, json.dumps({"error": str(error)}), "application/json")

    def log_message(self, format, *args):
        pass


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"Interfata ruleaza la {url}  (Ctrl+C pentru oprire)")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nOprit.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()