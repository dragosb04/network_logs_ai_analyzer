import json
import os
import re
import threading
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from ollama import ResponseError
from pydantic import BaseModel, Field

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
HOST = "127.0.0.1"
PORT = 8765

MAX_AI_ATTEMPTS = 2
CONTEXT_STEP = 3
MAX_CONTEXT_CHARS = 3000


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


def extract_error_context(lines, context_size):
    selected = set()
    for index, line in enumerate(lines):
        if "ERROR" in line:
            start = max(0, index - context_size)
            end = min(len(lines), index + context_size + 1)
            for i in range(start, end):
                selected.add(i)

    context_lines = []
    for i in sorted(selected):
        context_lines.append(lines[i])

    text = "\n".join(context_lines)
    return text[:MAX_CONTEXT_CHARS]


def build_ai_input(level_counts, error_messages, context_text):
    parts = [
        f"Errors: {level_counts['ERROR']}, Warnings: {level_counts['WARNING']}",
        "Most frequent errors:",
    ]
    for message, count in error_messages.most_common(10):
        parts.append(f"{count}x {message}")

    if context_text:
        parts.append("")
        parts.append("Raw log lines around the errors:")
        parts.append(context_text)

    return mask_sensitive("\n".join(parts))

class LogAnalysis(BaseModel):
    main_issue: str = Field(description="One sentence describing the main problem in the log. Never write the confidence here.")
    evidence: list[str] = Field(description="Log facts that support the main issue, e.g. error messages and their counts.")
    possible_causes: list[str]
    investigation_steps: list[str]
    confidence: Literal["High", "Medium", "Low"] = Field(description="How strongly the evidence points to the main issue.")


prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert system and network support engineer. Analyze this log summary carefully.\n"
     "Identify the core root cause, evaluate confidence, list likely causes, and actionable checks.\n"
      "Confidence rules:\n"
      "- High: most errors point to the same cause.\n"
      "- Medium: one cause is likely, but other errors could be related.\n"
      "- Low: errors point to several unrelated causes, or the information is too little.\n"
      "If there are several unrelated problems, pick the one with the most errors as main_issue.\n"
     "If raw log lines are included, use them to refine your answer."),
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

class AgentState(TypedDict):
    lines: list
    level_counts: Counter
    error_messages: Counter
    model: str
    context_size: int
    ai_input: str
    analysis: dict
    attempts: int


def build_input_node(state: AgentState) -> dict:
    context_text = ""
    if state["context_size"] > 0:
        context_text = extract_error_context(state["lines"], state["context_size"])

    ai_input = build_ai_input(state["level_counts"], state["error_messages"], context_text)
    return {"ai_input": ai_input}


def ask_ai_node(state: AgentState) -> dict:
    analysis = ask_ai(state["ai_input"], state["model"])
    attempts = state["attempts"] + 1
    return {"analysis": analysis, "attempts": attempts}


def more_context_node(state: AgentState) -> dict:
    new_size = state["context_size"] + CONTEXT_STEP
    return {"context_size": new_size}


def decide_next_step(state: AgentState) -> str:
    is_low = state["analysis"]["confidence"] == "Low"
    can_retry = state["attempts"] < MAX_AI_ATTEMPTS
    has_errors = state["level_counts"]["ERROR"] > 0

    if is_low and can_retry and has_errors:
        return "more_context"
    return "done"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("build_input", build_input_node)
    graph.add_node("ask_ai", ask_ai_node)
    graph.add_node("more_context", more_context_node)

    graph.add_edge(START, "build_input")
    graph.add_edge("build_input", "ask_ai")
    graph.add_conditional_edges(
        "ask_ai",
        decide_next_step,
        {"more_context": "more_context", "done": END},
    )
    graph.add_edge("more_context", "build_input")

    return graph.compile()


AGENT = build_graph()

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
        "attempts": 0,
        "context_lines": 0,
    }

    initial_state = {
        "lines": lines,
        "level_counts": level_counts,
        "error_messages": error_messages,
        "model": model,
        "context_size": 0,
        "ai_input": "",
        "analysis": {},
        "attempts": 0,
    }

    try:
        final_state = AGENT.invoke(initial_state)
        result["ai"] = final_state["analysis"]
        result["attempts"] = final_state["attempts"]
        result["context_lines"] = final_state["context_size"]
    except ConnectionError:
        result["ai_error"] = "Nu mă pot conecta la Ollama. Rulează? (încearcă: ollama serve)"
    except ResponseError:
        result["ai_error"] = f"Ollama a returnat o eroare. Modelul e descărcat? (încearcă: ollama pull {model})"
    except ValueError:
        result["ai_error"] = "Modelul nu a returnat un răspuns JSON valid."
    except Exception as error:
        result["ai_error"] = f"Analiza AI a eșuat: {error}"

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
  .badge-high { background: #dcfce7; color: #166534; }
  .badge-medium { background: #fef3c7; color: #92400e; }
  .badge-low { background: #e5e7eb; color: #374151; }
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
      <label id="drop" for="file">Trage un fișier .log aici sau dă click pentru a alege<input id="file" type="file" class="hidden"></label>
      <span id="fname" class="muted">Niciun fișier ales.</span>
    </div>
    <div class="row" style="margin-top:12px">
      <label>Model: <input id="model" type="text" readonly="readonly" value="__MODEL__"></label>
      <button id="go" disabled>Analizează</button>
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

    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <h2 style="margin:0;">Sugestie AI <span class="muted" id="ai-model"></span></h2>
        <span id="ai-badge" class="badge hidden"></span>
      </div>

      <div id="ai-content"></div>

      <p id="ai-agent" class="muted hidden" style="margin-top:14px; margin-bottom:0;"></p>
      <p class="muted" style="margin-top:8px; margin-bottom:0;">Un om trebuie să verifice această sugestie.</p>
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

function showMessage(text, className) {
  const aiContent = $("ai-content");
  aiContent.textContent = "";
  const span = document.createElement("span");
  if (className) span.className = className;
  span.textContent = text;
  aiContent.append(span);
}

function addListSection(box, title, items, listClass, numbered) {
  const section = document.createElement("div");
  section.className = "ai-section";

  const h3 = document.createElement("h3");
  h3.textContent = title;

  const ul = document.createElement("ul");
  ul.className = listClass;

  items.forEach((item, idx) => {
    const li = document.createElement("li");
    if (numbered) {
      const num = document.createElement("strong");
      num.textContent = (idx + 1) + ". ";
      li.append(num);
    }
    li.append(document.createTextNode(item));
    ul.append(li);
  });

  section.append(h3, ul);
  box.append(section);
}

function showBadge(confidence) {
  const aiBadge = $("ai-badge");
  const level = confidence.toLowerCase();

  let badgeClass = "badge-medium";
  if (level === "high") badgeClass = "badge-high";
  if (level === "low") badgeClass = "badge-low";

  aiBadge.textContent = "Certitudine: " + confidence;
  aiBadge.className = "badge " + badgeClass;
}

function showAgentInfo(attempts, contextLines) {
  const info = $("ai-agent");
  if (attempts > 1) {
    info.textContent = "Agentul a avut certitudine scăzută și a reîncercat cu ±" + contextLines
      + " linii de context în jurul erorilor (" + attempts + " încercări).";
  } else {
    info.textContent = "Răspuns obținut din prima încercare.";
  }
  info.classList.remove("hidden");
}

function showAnalysis(aiData) {
  const box = document.createElement("div");
  box.className = "ai-box";

  if (aiData.main_issue) {
    const issue = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = "Problema identificată:";
    const p = document.createElement("p");
    p.style.margin = "4px 0 0";
    p.textContent = aiData.main_issue;
    issue.append(title, p);
    box.append(issue);
  }

  if (aiData.possible_causes && aiData.possible_causes.length) {
    addListSection(box, "Cauze posibile", aiData.possible_causes, "ai-list", false);
  }

  if (aiData.investigation_steps && aiData.investigation_steps.length) {
    addListSection(box, "Ce ar trebui verificat", aiData.investigation_steps, "steps-list", true);
  }

  $("ai-content").textContent = "";
  $("ai-content").append(box);
}

$("go").addEventListener("click", async () => {
  $("go").disabled = true;
  $("out").classList.remove("hidden");
  $("ai-badge").className = "badge hidden";
  $("ai-agent").classList.add("hidden");
  $("ai-model").textContent = "";
  showMessage("Se așteaptă răspunsul de la model...", "muted");

  try {
    const res = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: logText, model: $("model").value.trim() })
    });
    const d = await res.json();
    if (!res.ok) {
      throw new Error(d.error || "Serverul a returnat o eroare.");
    }

    $("s-total").textContent = d.total;
    $("s-error").textContent = d.error;
    $("s-warning").textContent = d.warning;
    $("s-info").textContent = d.info;

    const t = $("errs");
    t.textContent = "";
    if (!d.top_errors.length) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.className = "muted";
      td.textContent = "Nu au fost găsite erori.";
      tr.append(td);
      t.append(tr);
    }
    d.top_errors.forEach(e => {
      const tr = document.createElement("tr");
      const a = document.createElement("td"); a.className = "n"; a.textContent = e.count + "x";
      const b = document.createElement("td"); b.textContent = e.message;
      tr.append(a, b);
      t.append(tr);
    });

    if (d.ai) {
      $("ai-model").textContent = "(" + d.model + ")";
      if (d.ai.confidence) showBadge(d.ai.confidence);
      showAnalysis(d.ai);
      showAgentInfo(d.attempts, d.context_lines);
    } else {
      showMessage(d.ai_error, "error");
    }
  } catch (err) {
    showMessage("Eroare: " + err.message, "error");
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
    print(f"Interfața rulează la {url}  (Ctrl+C pentru oprire)")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nOprit.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()