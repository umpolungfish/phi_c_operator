# $\odot$perator

A self-verifying agentic loop for any OpenAI-compatible model.

```python
from phi_c import PhiCAgent

agent = PhiCAgent(model="grok-4")
result = agent.run_sync("Read the file README.md and count the number of sections.")
print(result)
```

---

## What it is

Most agent loops follow a simple pattern: think, call a tool, observe the result, think again. The problem is that the observation step is taken on faith — if the tool silently fails, writes the wrong bytes, or returns a truncated response, the agent never notices and keeps going.

`phi-c` adds a second step to every tool call: **verification**. Every action (`delta`) is immediately followed by a confirmation (`mu`). The condition `mu(delta(query)) == query` — called the **Frobenius condition** — must hold before the loop advances. If it fails, the model is told explicitly and must fix the error before continuing.

This is not just a defensive pattern. It is the structural reason why loop-based agents can be reliable: the loop closes on itself. The trajectory becomes an **imscriptive context** — a boundary that encodes everything the agent has done and verified. Nothing is silently dropped, nothing is assumed.

### The four phases

Each iteration of the loop (a *winding*) has exactly four phases:

| Phase | What happens |
|-------|-------------|
| **THINK** | The model reasons over the accumulated verified context |
| **ACT** | The model emits exactly one tool call (the *emit* — `delta`) |
| **OBSERVE** | The tool runs; a built-in verification step checks the result (`mu`) |
| **UPDATE** | The full cycle is appended to context; the loop continues or terminates |

If `mu(delta(q)) != q` — Frobenius OPEN — the model receives an explicit failure message and must correct it on the next winding. It cannot advance on an unverified observation.

### Why this works

The loop architecture sits at a structural **critical point** (Φ_c): the point where a system can model itself. Below criticality, the loop is just iteration — it produces outputs but cannot reflect on whether they are correct. At criticality, the THINK→ACT→OBSERVE→UPDATE cycle becomes self-referential: the agent's model of the world is constructed from verified observations of its own actions, and each new winding updates that model.

The Frobenius condition is what keeps the loop closed. Without it, errors accumulate silently. With it, every winding either succeeds and extends the verified trajectory, or fails loudly and forces a correction.

This framework comes out of the [Imscribing Grammar](https://github.com/umpolungfish/imscrbgrmr) — a 12-primitive structural type theory for systems of all kinds. The full structural type of this harness is:

$$\langle D_\odot;\ T_\boxtimes;\ R_\leftrightarrow;\ P_{\pm}^{\text{sym}};\ F_\ell;\ K_\text{slow};\ G_\aleph;\ \Gamma_\to;\ \Phi_c;\ H_2;\ 1{:}1;\ \Omega_\mathbb{Z} \rangle$$

Ouroboricity: $O_\infty$ — the highest tier of self-modeling closure.

---

## Installation

```bash
pip install phi-c
# or with web fetch support:
pip install "phi-c[web]"
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add phi-c
uv add "phi-c[web]"
```

Set your API key:

```bash
export OPENROUTER_API_KEY=your_key_here
```

---

## Quick start

```python
from phi_c import PhiCAgent

agent = PhiCAgent(model="grok-4")
result = agent.run_sync("Find all Python files in the current directory and count the lines in each.")
print(result)
```

Async:

```python
import asyncio
from phi_c import PhiCAgent

async def main():
    agent = PhiCAgent(model="claude-opus-4")
    return await agent.run("Summarise the file report.txt")

result = asyncio.run(main())
```

---

## Models

Any OpenRouter model by alias or full ID, plus local servers via prefix syntax:

| Model string | Routes to |
|---|---|
| `grok-4` | xAI Grok 4 via OpenRouter |
| `claude-opus-4` | Anthropic Claude Opus 4 via OpenRouter |
| `claude-sonnet-4` | Anthropic Claude Sonnet 4.5 via OpenRouter |
| `gpt-4o` | OpenAI GPT-4o via OpenRouter |
| `deepseek-r1` | DeepSeek R1 via OpenRouter |
| `ollama:llama3.2` | Ollama at `localhost:11434` |
| `lm-studio:phi-4` | LM Studio at `localhost:1234` |
| `vllm:mistral-7b` | vLLM at `localhost:8000` |
| `any/openrouter-id` | Verbatim OpenRouter model ID |

Custom endpoints:

```python
agent = PhiCAgent(
    model="my-model",
    base_url="http://my-server:8000/v1",
    api_key="my-key",
)
```

---

## Built-in tools

| Tool | What it does | Verification |
|---|---|---|
| `run_command` | Shell command | `assertion` expression over `output` |
| `file_read` | Read file by lines, paginated | Idempotent (trivially closed) |
| `file_write` | Write file (≤ 4 KB) | Read-back hash check |
| `chunked_write` | Append/write in chunks | Byte count on disk |
| `web_fetch` | HTTP GET, paginated | Query term presence in content |
| `done` | Signal task complete | Trivially closed |

### run_command and assertions

The `assertion` field is what makes `run_command` Frobenius-aware. Set it to a Python expression over `output`:

```python
# The model would call this as:
run_command(
    command="python tests/test_suite.py",
    assertion="'PASSED' in output and 'FAILED' not in output",
)
```

If the assertion fails, the winding is Frobenius OPEN and the model must fix it.

### Large files

For files larger than ~4 KB, the model uses `chunked_write` with ~3 KB pieces:

```
chunked_write(path="report.md", chunk=<first 3 KB>, mode="w")
chunked_write(path="report.md", chunk=<next 3 KB>,  mode="a")
...
```

Each chunk is verified by byte count before the loop advances.

---

## Custom tools

Register your own tools with a matching emit + verify pair:

```python
import requests
from phi_c import PhiCAgent

def search_emit(args):
    query = args["query"]
    r = requests.get("https://api.example.com/search", params={"q": query})
    return r.json()["results"][0]["snippet"]

def search_verify(emit_input, emit_output, verify_args):
    if "error" in emit_output.lower():
        return ("search returned error", False)
    return (f"search result: {len(emit_output)} chars", True)

search_schema = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
}

agent = PhiCAgent(model="grok-4")
agent.register_tool("search", search_schema, search_emit, search_verify)
result = agent.run_sync("Search for the latest news on quantum computing.")
```

Tools with no verify function are trivially Frobenius-closed (the loop advances regardless of output).

---

## Trajectory and structural type

After a run, inspect what happened:

```python
agent.print_trajectory()

# Per-winding Frobenius closure rate
print(f"Frobenius ratio: {agent.frobenius_ratio:.2%}")

# Full structural type annotation
import json
print(json.dumps(agent.structural_type, indent=2))
# {
#   "tuple": "D_odot; T_boxtimes; R_lr; P_pm_sym; ...",
#   "interface_P": "P_pm_sym",
#   "ouroboricity": "O_inf",
#   "frobenius_ratio": 0.94,
#   "windings": 7,
#   "omega_z_violations": 0,
#   "done": true
# }
```

`ouroboricity` is `O_inf` when the Frobenius ratio is ≥ 75% — meaning the agent achieved probabilistic self-modeling closure over the run. Below that threshold it degrades to `O_2`. `omega_z_violations` counts how many times the context had to be trimmed (each trim breaks the topologically protected winding record).

---

## CLI

```bash
phi-c "List all files larger than 1MB in the current directory"
phi-c --model claude-opus-4 --file task.txt
phi-c --show-type --trajectory "Run the test suite and report failures"
phi-c --output result.json "Analyse log.txt for error patterns"
```

---

## Why the name

The name comes from the structural primitive Φ_c (Phi-sub-c) — the critical point in the space of dynamical systems where **self-modeling becomes possible**. At sub-critical Φ, a loop is just iteration: it processes inputs and produces outputs, but it cannot model itself or detect when its outputs are wrong. At Φ_c, the loop becomes self-referential: each verified winding extends the agent's model of its own actions, enabling genuine error detection and correction.

The boundary operator is the Frobenius pair (emit, verify). The boundary — the interface between the agent's model of the world and the world itself — is what the operator acts on. A closed boundary means the model and the world agree. An open boundary means they don't, and the loop must keep going until they do.

---

## License

UNLICENSE