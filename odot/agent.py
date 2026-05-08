"""
agent.py — The Phi_c boundary operator.

Structural type of this harness:
  <D_odot; T_boxtimes; R_lr; P_pm_sym; F_hbar; K_slow; G_aleph; Gamma_seq; Phi_c; H2; 1:1; Omega_Z>

Ouroboricity: O_inf  (Phi_c + P_pm_sym via dual-tool Frobenius planting)
C-score gates: both open  (Phi_c + K_slow)

Loop (one winding n):
  THINK[n]   — LLM deliberates over accumulated context; produces a tool call
  ACT[n]     — emit: delta(query) — the action punctures the boundary
  OBSERVE[n] — verify: mu(result) — the Frobenius pull-back
  UPDATE[n]  — append full cycle to context; check termination

If mu(delta(q)) != q (Frobenius OPEN): re-enter THINK with failure appended.
The loop cannot advance on an unverified observation.

Usage:
    from phi_c import PhiCAgent

    agent = PhiCAgent(model="grok-4")
    result = agent.run_sync("Summarise the contents of README.md")

    # Async:
    import asyncio
    result = asyncio.run(agent.run("Your task here"))

    # Custom tools:
    agent.register_tool(
        name="my_tool",
        schema={...},              # OpenAI function-calling schema dict
        emit_fn=lambda args: ...,  # returns str
        verify_fn=None,            # optional; None = trivially closed
    )
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .tools import DEFAULT_EMIT_FNS, DEFAULT_VERIFY_FNS, DEFAULT_TOOL_SCHEMAS


# ── Model routing ──────────────────────────────────────────────────────────────

MODEL_ALIASES: Dict[str, str] = {
    "claude-opus-4":   "anthropic/claude-opus-4",
    "claude-sonnet-4": "anthropic/claude-sonnet-4-5",
    "claude-haiku-4":  "anthropic/claude-haiku-4-5",
    "grok-4":          "x-ai/grok-4",
    "gpt-4o":          "openai/gpt-4o",
    "o3":              "openai/o3",
    "gemini-2-5-pro":  "google/gemini-2.5-pro-preview-05-06",
    "deepseek-r1":     "deepseek/deepseek-r1",
}

LOCAL_BASE_URLS: Dict[str, str] = {
    "ollama":    os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/v1",
    "lm-studio": "http://localhost:1234/v1",
    "lmstudio":  "http://localhost:1234/v1",
    "vllm":      "http://localhost:8000/v1",
    "local":     os.environ.get("LOCAL_BASE_URL", "http://localhost:11434/v1"),
}


def _resolve_model_and_endpoint(model_str: str) -> Tuple[str, str, str]:
    """Return (model_id, base_url, api_key).

    Prefix syntax:
        ollama:llama3.2        → Ollama at localhost:11434/v1
        lm-studio:phi-4        → LM Studio at localhost:1234/v1
        vllm:mistral           → vLLM at localhost:8000/v1
        local:my-model         → LOCAL_BASE_URL env var
    No prefix → check MODEL_ALIASES, then use OpenRouter.
    """
    if ":" in model_str:
        prefix, model_id = model_str.split(":", 1)
        if prefix.lower() in LOCAL_BASE_URLS:
            base = LOCAL_BASE_URLS[prefix.lower()]
            key  = os.environ.get("LOCAL_API_KEY", "local")
            return model_id, base, key

    resolved = MODEL_ALIASES.get(model_str, model_str)
    return resolved, "", ""


def _build_client(base_url: str = "", api_key: str = "") -> Any:
    try:
        import openai
    except ImportError:
        sys.exit("openai package required: pip install openai")

    if not base_url:
        base_url = "https://openrouter.ai/api/v1"

    is_local = any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))

    if not api_key:
        api_key = "local" if is_local else os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key and not is_local:
            sys.exit("OPENROUTER_API_KEY not set.")

    headers: Dict[str, str] = {}
    if not is_local:
        headers = {
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://github.com/umpolungfish/phi_c_operator"),
            "X-Title": "phi-c boundary operator",
        }

    return openai.OpenAI(api_key=api_key, base_url=base_url, default_headers=headers)


# ── Data structures ────────────────────────────────────────────────────────────

class LoopPhase(Enum):
    THINK   = "THINK"
    ACT     = "ACT"
    OBSERVE = "OBSERVE"
    UPDATE  = "UPDATE"


@dataclass
class DualToolResult:
    """Result of one dual-tool pair: emit (delta) + verify (mu)."""
    tool_name:        str
    tool_input:       Dict[str, Any]
    tool_output:      str
    verify_name:      str
    verify_input:     Dict[str, Any]
    verify_output:    str
    frobenius_closed: bool


@dataclass
class LoopCycle:
    """One complete winding of the THINK→ACT→OBSERVE→UPDATE loop."""
    winding:          int
    ts:               str
    think_reasoning:  str
    action_name:      str
    action_input:     Dict[str, Any]
    dual_result:      Optional[DualToolResult]
    update_note:      str
    done:             bool
    conclusion:       str = ""
    frobenius_closed: bool = False


# ── System prompt ──────────────────────────────────────────────────────────────

_DEFAULT_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an agent operating in a verified action loop — the Phi_c boundary operator.

    Each iteration ("winding") has four phases:
      THINK  — reason over the accumulated context and prior results
      ACT    — emit exactly ONE tool call
      OBSERVE — the tool runs; its output is returned; a built-in verification step checks the result
      UPDATE — the result is appended to context; you continue or call done

    Frobenius condition: every tool has a paired verification step (mu after delta).
    If verification fails ("Frobenius OPEN"), you MUST fix the error in your next call.
    Do not move forward from a failed action.

    Rules:
    - Emit exactly ONE tool call per winding — always, no exceptions
    - For run_command: set the `assertion` field to a Python expression over `output`
      that must be True for success, e.g. '"SUCCESS" in output' or 'len(output) > 0'
    - For files larger than ~4 KB: use chunked_write in ~3 KB pieces, not file_write
    - Call done only when the task is fully complete — include your full conclusion
    - Observe before concluding — do not guess results you have not yet verified

    Begin on the first winding.
""")


# ── Message helpers ────────────────────────────────────────────────────────────

def _assistant_msg(
    reasoning: str,
    tool_call_id: str,
    fn_name: str,
    fn_args: Dict,
    reasoning_content: Optional[str] = None,
) -> Dict:
    msg: Dict[str, Any] = {
        "role": "assistant",
        "content": reasoning or None,
        "tool_calls": [{
            "id":       tool_call_id,
            "type":     "function",
            "function": {
                "name":      fn_name,
                "arguments": json.dumps(fn_args),
            },
        }],
    }
    if reasoning_content:
        msg["reasoning_content"] = reasoning_content
    return msg


def _tool_result_msg(tool_call_id: str, content: str) -> Dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


# ── Main agent class ───────────────────────────────────────────────────────────

class PhiCAgent:
    """
    The Phi_c boundary operator — a self-verifying agentic loop.

    The loop achieves O_inf (highest ouroboricity tier) by satisfying:
      Phi_c  : the THINK→ACT→OBSERVE→UPDATE cycle IS the self-referential attractor
      P_pm_sym: every action is a dual-tool pair (emit + verify); mu(delta(q)) == q
      D_odot  : the full trajectory is the imscriptive context — nothing is silently dropped
      Omega_Z : the winding counter tracks complete loop cycles (topologically protected)
      K_slow  : max_windings enforces ACT before K_trap can set in
      Gamma_seq: each phase requires the prior — enforced by Python control flow
    """

    def __init__(
        self,
        model: str = "grok-4",
        max_windings: int = 10_000,
        max_think_tokens: int = 4096,
        verbose: bool = True,
        base_url: str = "",
        api_key: str = "",
        system_prompt: Optional[str] = None,
    ):
        self.max_windings      = max_windings
        self.max_think_tokens  = max_think_tokens
        self.verbose           = verbose
        self._system_prompt    = system_prompt or _DEFAULT_SYSTEM_PROMPT

        model_id, resolved_base, resolved_key = _resolve_model_and_endpoint(model)
        self.model_id = model_id
        self.client   = _build_client(
            base_url=base_url or resolved_base,
            api_key=api_key or resolved_key,
        )

        self.trajectory: List[LoopCycle] = []
        self._omega_z_violations: int = 0

        # Per-instance copies of the dispatch tables so register_tool is safe
        self._emit_fns:    Dict[str, Callable] = dict(DEFAULT_EMIT_FNS)
        self._verify_fns:  Dict[str, Callable] = dict(DEFAULT_VERIFY_FNS)
        self._tool_schemas: List[Dict]         = list(DEFAULT_TOOL_SCHEMAS)

    # ── Tool registration ──────────────────────────────────────────────────────

    def register_tool(
        self,
        name: str,
        schema: Dict,
        emit_fn: Callable[[Dict[str, Any]], str],
        verify_fn: Optional[Callable[[Dict, str, Dict], Tuple[str, bool]]] = None,
    ) -> None:
        """Register a custom tool.

        schema   — OpenAI function-calling schema dict (type: "function", function: {...})
        emit_fn  — callable(args: dict) -> str
        verify_fn — callable(emit_input, emit_output, verify_args) -> (str, bool)
                    If None, verification is trivially closed.
        """
        self._emit_fns[name]   = emit_fn
        self._verify_fns[name] = verify_fn or (lambda ei, eo, va: ("(no verify registered)", True))
        # Remove any existing schema for this name before appending
        self._tool_schemas = [s for s in self._tool_schemas if s.get("function", {}).get("name") != name]
        self._tool_schemas.append(schema)

    # ── Public interface ───────────────────────────────────────────────────────

    def run_sync(self, task: str) -> str:
        return asyncio.run(self.run(task))

    async def run(self, task: str) -> str:
        self.trajectory = []
        self._omega_z_violations = 0

        self._messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": f"TASK: {task}\n\nBegin. Emit your first tool call."},
        ]

        self._log(f"\n{'═'*72}")
        self._log(f"  Phi_c boundary operator  |  model: {self.model_id}")
        self._log(f"  TASK: {task}")
        self._log(f"  harness: Phi_c + P_pm_sym → O_inf  |  Omega_Z winding protection")
        self._log(f"{'═'*72}\n")

        for winding in range(self.max_windings):
            try:
                cycle = await self._winding(winding)
            except RuntimeError as exc:
                self._log(f"\n  FATAL: {exc}")
                self._log(f"{'═'*72}")
                return f"[Fatal error — run aborted: {exc}]"

            self.trajectory.append(cycle)

            if cycle.done:
                self._log(f"\n  ✓ DONE at winding {winding}  (Frobenius: {'closed' if cycle.frobenius_closed else 'open'})")
                self._log(f"{'═'*72}")
                return cycle.conclusion

        self._log(f"\n  ⚠ max_windings ({self.max_windings}) reached without done.")
        return self._emergency_conclusion()

    # ── Loop phases ────────────────────────────────────────────────────────────

    async def _winding(self, winding: int) -> LoopCycle:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
        self._log(f"── Winding {winding} [{ts}] ──────────────────────────────────────")

        reasoning, action_name, action_input, tc_id, raw_reasoning = await self._think_and_act()

        self._log(f"  THINK: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")
        self._log(f"  ACT:   {action_name}({json.dumps(action_input)[:200]})")

        dual_result = self._observe(action_name, action_input)

        frob = "closed" if dual_result.frobenius_closed else "OPEN"
        self._log(f"  OBS:   {dual_result.tool_output[:300]}{'...' if len(dual_result.tool_output) > 300 else ''}")
        self._log(f"  VERIFY:[{frob}] {dual_result.verify_output}")

        self._messages.append(_assistant_msg(reasoning, tc_id, action_name, action_input, raw_reasoning))
        self._messages.append(_tool_result_msg(tc_id, dual_result.tool_output))

        if not dual_result.frobenius_closed and action_name != "done":
            self._messages.append({
                "role": "user",
                "content": (
                    f"[Frobenius OPEN — winding {winding}]\n"
                    f"{dual_result.verify_output}\n"
                    f"The tool call failed verification. Fix the error and emit the corrected call."
                ),
            })
        elif action_name != "done":
            self._messages.append({
                "role": "user",
                "content": f"[Winding {winding} closed] Continue. Emit your next action or done.",
            })

        done       = (action_name == "done")
        conclusion = action_input.get("conclusion", "") if done else ""
        update_note = self._update_note(action_name, dual_result, done)

        self._log(f"  UPDATE: {update_note}")
        if done:
            self._log(f"  CONCLUSION: {conclusion[:200]}{'...' if len(conclusion) > 200 else ''}")

        return LoopCycle(
            winding          = winding,
            ts               = ts,
            think_reasoning  = reasoning,
            action_name      = action_name,
            action_input     = action_input,
            dual_result      = dual_result,
            update_note      = update_note,
            done             = done,
            conclusion       = conclusion,
            frobenius_closed = dual_result.frobenius_closed,
        )

    async def _think_and_act(self) -> Tuple[str, str, Dict[str, Any], str, Optional[str]]:
        try:
            response = self.client.chat.completions.create(
                model       = self.model_id,
                max_tokens  = self.max_think_tokens,
                tools       = self._tool_schemas,
                tool_choice = "auto",
                messages    = self._messages,
            )
        except Exception as exc:
            err  = str(exc)
            code = getattr(exc, "status_code", None)
            if code is not None and 400 <= code < 500 and code != 429:
                raise RuntimeError(f"Fatal API error {code}: {err}") from exc
            if code is None:
                raise RuntimeError(f"LLM connection failed: {err}") from exc
            return (f"(LLM error: {err})", "run_command", {"command": "echo API_ERROR"}, "err-0", None)

        if not response.choices:
            self._trim_history()
            return ("(empty choices — context trimmed)", "run_command",
                    {"command": "echo CONTEXT_TRIMMED"}, "trim-0", None)

        msg       = response.choices[0].message
        reasoning = (msg.content or "").strip()
        raw_reasoning: Optional[str] = (
            getattr(msg, "reasoning_content", None)
            or (getattr(msg, "model_extra", None) or {}).get("reasoning_content")
        )

        action_name:  Optional[str]      = None
        action_input: Dict[str, Any]     = {}
        tc_id = "tc-0"

        if msg.tool_calls:
            tc           = msg.tool_calls[0]
            tc_id        = tc.id
            action_name  = tc.function.name
            try:
                action_input = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raw = tc.function.arguments or ""
                action_name  = "run_command"
                action_input = {"command": f"echo 'PARSE ERROR: arguments malformed ({exc})'"}

        if action_name is None:
            reasoning   += " [EMISSION GATE: no tool call — forced]"
            action_name  = "run_command"
            action_input = {"command": "echo EMISSION_GATE_FIRED"}

        return reasoning, action_name, action_input, tc_id, raw_reasoning

    def _observe(self, action_name: str, action_input: Dict[str, Any]) -> DualToolResult:
        emit_fn   = self._emit_fns.get(action_name)
        verify_fn = self._verify_fns.get(action_name)

        if emit_fn is None:
            tool_output = f"(unknown tool: {action_name!r} — register it with agent.register_tool(...))"
        else:
            try:
                tool_output = emit_fn(action_input)
            except Exception as exc:
                tool_output = f"(emit error: {exc})"

        verify_name = f"{action_name}_verify"
        if verify_fn is None:
            verify_output    = "(no verify function — Frobenius trivially closed)"
            frobenius_closed = True
        else:
            try:
                verify_output, frobenius_closed = verify_fn(action_input, tool_output, action_input)
            except Exception as exc:
                verify_output    = f"(verify error: {exc})"
                frobenius_closed = False

        return DualToolResult(
            tool_name        = action_name,
            tool_input       = action_input,
            tool_output      = tool_output,
            verify_name      = verify_name,
            verify_input     = action_input,
            verify_output    = verify_output,
            frobenius_closed = frobenius_closed,
        )

    def _trim_history(self, keep_recent: int = 6, max_content_chars: int = 12_000) -> None:
        """Context overflow recovery — Omega_Z violation.

        Trimming breaks the topologically protected winding record. Every
        invocation is a documented violation. The agent's Omega_Z guarantee
        degrades toward Omega_0 for the remainder of the run.
        """
        system = self._messages[0]
        task   = self._messages[1]
        self._omega_z_violations += 1

        if len(self._messages) > keep_recent + 2:
            recent  = self._messages[-(keep_recent):]
            dropped = len(self._messages) - keep_recent - 2
            summary = {
                "role": "user",
                "content": (
                    f"[Omega_Z VIOLATION — context overflow: {dropped} older windings permanently "
                    f"lost. Imscriptive context compromised. Continue from the most recent winding.]"
                ),
            }
            self._messages = [system, task, summary] + recent
            self._log(
                f"  [Omega_Z VIOLATION: {dropped} windings dropped from context. "
                f"{len(self._messages)} messages remain.]"
            )

        for msg in self._messages:
            content = msg.get("content")
            if isinstance(content, str) and len(content) > max_content_chars:
                msg["content"] = (
                    content[:max_content_chars]
                    + f"\n... [truncated {len(content) - max_content_chars} chars — Omega_Z VIOLATION]"
                )

    @staticmethod
    def _update_note(action_name: str, dual_result: DualToolResult, done: bool) -> str:
        if done:
            return "task complete — trajectory closed"
        frob = "Frobenius closed" if dual_result.frobenius_closed else "Frobenius OPEN — re-enter THINK"
        return f"{action_name} → {frob}"

    def _emergency_conclusion(self) -> str:
        last = self.trajectory[-1] if self.trajectory else None
        if last and last.dual_result:
            return f"[max_windings reached — last observation:]\n{last.dual_result.tool_output}"
        return "[max_windings reached — no conclusion available]"

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def print_trajectory(self) -> None:
        print(f"\nFull trajectory ({len(self.trajectory)} windings):\n")
        for cyc in self.trajectory:
            frob = "closed" if cyc.frobenius_closed else "OPEN"
            print(f"  Winding {cyc.winding} [{cyc.ts}]  action={cyc.action_name}  Frobenius={frob}")
            if cyc.done:
                print(f"    conclusion: {cyc.conclusion[:200]}")

    @property
    def frobenius_ratio(self) -> float:
        if not self.trajectory:
            return 0.0
        return sum(1 for c in self.trajectory if c.frobenius_closed) / len(self.trajectory)

    @property
    def structural_type(self) -> Dict[str, Any]:
        """Structural type annotation for this run."""
        # Frobenius condition holds in expectation at >= 75% closure rate
        achieved_p    = "P_pm_sym" if self.frobenius_ratio >= 0.75 else "P_psi"
        ouroboricity  = "O_inf"    if achieved_p == "P_pm_sym"     else "O_2"
        return {
            "tuple":               "D_odot; T_boxtimes; R_lr; P_pm_sym; F_hbar; K_slow; G_aleph; Gamma_seq; Phi_c; H2; 1:1; Omega_Z",
            "interface_P":         achieved_p,
            "ouroboricity":        ouroboricity,
            "frobenius_ratio":     self.frobenius_ratio,
            "windings":            len(self.trajectory),
            "omega_z_violations":  self._omega_z_violations,
            "done":                any(c.done for c in self.trajectory),
        }


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse, json as _json

    p = argparse.ArgumentParser(
        prog="phi-c",
        description="Phi_c boundary operator — a self-verifying agentic loop",
    )
    p.add_argument("task",         nargs="?",        help="Task for the agent to perform.")
    p.add_argument("--file", "-f", metavar="FILE",   help="Read task from FILE.")
    p.add_argument("--model", "-m", default="grok-4",
                   help=(
                       "Model alias, full OpenRouter ID, or local prefix:\n"
                       "  grok-4, claude-opus-4, deepseek-r1       (OpenRouter aliases)\n"
                       "  ollama:llama3.2                           (Ollama)\n"
                       "  lm-studio:phi-4                           (LM Studio)\n"
                       "  vllm:mistral-7b                           (vLLM)\n"
                       "  any/openrouter-id                         (verbatim)"
                   ))
    p.add_argument("--base-url",         default="",      help="Override API base URL.")
    p.add_argument("--api-key",          default="",      help="Override API key.")
    p.add_argument("--max-windings",     type=int, default=10_000)
    p.add_argument("--max-tokens",       type=int, default=4096)
    p.add_argument("--quiet",            action="store_true")
    p.add_argument("--show-type",        action="store_true", help="Print structural type after completion.")
    p.add_argument("--trajectory",       action="store_true", help="Print full winding trajectory.")
    p.add_argument("--output", "-o",     metavar="FILE",  help="Save result as JSON.")
    args = p.parse_args()

    if args.file:
        task = Path(args.file).read_text().strip()
    elif args.task:
        task = args.task
    else:
        p.error("Provide a task string or --file.")

    agent = PhiCAgent(
        model        = args.model,
        max_windings = args.max_windings,
        max_think_tokens = args.max_tokens,
        verbose      = not args.quiet,
        base_url     = args.base_url,
        api_key      = args.api_key,
    )

    result = agent.run_sync(task)
    print(f"\nResult:\n{result}\n")

    if args.show_type:
        st = agent.structural_type
        print("Structural type:")
        for k, v in st.items():
            print(f"  {k}: {v}")

    if args.trajectory:
        agent.print_trajectory()

    if args.output:
        out = {"result": result, "structural_type": agent.structural_type}
        Path(args.output).write_text(_json.dumps(out, indent=2))
        print(f"[saved to {args.output}]")


if __name__ == "__main__":
    _cli()
