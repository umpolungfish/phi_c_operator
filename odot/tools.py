"""
tools.py — Universal dual-tool battery for the Phi_c boundary operator.

Each tool is a (emit_fn, verify_fn) pair.
  emit_fn(args: dict) -> str          — execute the action
  verify_fn(emit_input, emit_output, verify_args) -> (str, bool)
                                       — verify Frobenius closure

The Frobenius condition: mu(delta(query)) == query.
emit is delta (boundary puncture); verify is mu (pull-back).
A closed pair means the observation round-trips faithfully.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fn(name: str, description: str, properties: Dict, required: List[str]) -> Dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _str(description: str, default: Optional[str] = None) -> Dict:
    d: Dict = {"type": "string", "description": description}
    if default is not None:
        d["default"] = default
    return d


def _int(description: str, default: Optional[int] = None) -> Dict:
    d: Dict = {"type": "integer", "description": description}
    if default is not None:
        d["default"] = default
    return d


# ── run_command ────────────────────────────────────────────────────────────────

def _run_command_emit(args: Dict[str, Any]) -> str:
    cmd     = args["command"]
    timeout = int(args.get("timeout", 30))
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        out = r.stdout + r.stderr
        return out if out.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return f"(timeout after {timeout}s)"
    except Exception as exc:
        return f"(error: {exc})"


def _run_command_verify(
    emit_input: Dict, emit_output: str, verify_args: Dict
) -> Tuple[str, bool]:
    assertion = verify_args.get("assertion", "")
    if not assertion:
        return ("(no assertion — Frobenius trivially closed)", True)
    ns = {"output": emit_output, "out": emit_output}
    try:
        ok = bool(eval(assertion, {"__builtins__": {}}, ns))  # noqa: S307
    except Exception as exc:
        return (f"assertion eval error: {exc}", False)
    if ok:
        return (f"assertion '{assertion}' PASSED", True)
    return (f"assertion '{assertion}' FAILED", False)


# ── file_read ──────────────────────────────────────────────────────────────────

def _file_read_emit(args: Dict[str, Any]) -> str:
    path   = args["path"]
    offset = int(args.get("offset", 0))
    limit  = int(args.get("limit", 200))
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        total = len(lines)
        chunk = lines[offset: offset + limit]
        header = f"[{path} — lines {offset+1}–{min(offset+limit, total)} of {total}]\n"
        if offset + limit < total:
            header += f"[use offset={offset+limit} to continue]\n"
        return header + "\n".join(chunk)
    except Exception as exc:
        return f"(error reading {path}: {exc})"


def _file_read_verify(
    emit_input: Dict, emit_output: str, verify_args: Dict
) -> Tuple[str, bool]:
    return ("(read is idempotent — Frobenius trivially closed)", True)


# ── file_write ─────────────────────────────────────────────────────────────────

def _file_write_emit(args: Dict[str, Any]) -> str:
    if "path" not in args or "content" not in args:
        missing = [k for k in ("path", "content") if k not in args]
        return f"(file_write error: missing required arg(s): {missing})"
    path, content = args["path"], args["content"]
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        digest = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"written {len(content)} bytes to {path}  (sha256:{digest})"
    except Exception as exc:
        return f"(error writing {path}: {exc})"


def _file_write_verify(
    emit_input: Dict, emit_output: str, verify_args: Dict
) -> Tuple[str, bool]:
    path     = emit_input.get("path", "")
    original = emit_input.get("content", "")
    try:
        readback = Path(path).read_text(encoding="utf-8")
        if readback == original:
            digest = hashlib.sha256(readback.encode()).hexdigest()[:16]
            return (f"read-back matches written content (sha256:{digest})", True)
        return (f"read-back MISMATCH — {len(readback)} vs {len(original)} chars", False)
    except Exception as exc:
        return (f"read-back error: {exc}", False)


# ── chunked_write ──────────────────────────────────────────────────────────────

def _chunked_write_emit(args: Dict[str, Any]) -> str:
    missing = [k for k in ("path", "chunk") if k not in args]
    if missing:
        return f"(chunked_write error: missing required arg(s): {missing})"
    path  = args["path"]
    chunk = args["chunk"]
    mode  = args.get("mode", "a")
    if mode not in ("w", "a"):
        return f"(chunked_write error: mode must be 'w' or 'a', got {mode!r})"
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open(mode, encoding="utf-8") as fh:
            fh.write(chunk)
        total  = p.stat().st_size
        digest = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        return f"wrote {len(chunk)} chars (mode={mode!r}); file total {total} bytes  (sha256:{digest})"
    except Exception as exc:
        return f"(chunked_write error to {path}: {exc})"


def _chunked_write_verify(
    emit_input: Dict, emit_output: str, verify_args: Dict
) -> Tuple[str, bool]:
    path = emit_input.get("path", "")
    try:
        size = Path(path).stat().st_size
        ok   = "error" not in emit_output.lower()
        return (f"{path}: {size} bytes on disk", ok)
    except Exception as exc:
        return (f"verify error: {exc}", False)


# ── web_fetch ──────────────────────────────────────────────────────────────────

def _web_fetch_emit(args: Dict[str, Any]) -> str:
    url         = args["url"]
    start_index = int(args.get("start_index", 0))
    max_chars   = int(args.get("max_chars", 8000))
    try:
        import httpx
        r = httpx.get(
            url, timeout=15, follow_redirects=True,
            headers={"User-Agent": "phi-c-agent/0.1"},
        )
        r.raise_for_status()
        text  = r.text
        total = len(text)
        chunk = text[start_index: start_index + max_chars]
        header = f"[{url} — chars {start_index}–{min(start_index + max_chars, total)} of {total}]\n"
        if start_index + max_chars < total:
            header += f"[use start_index={start_index + max_chars} to continue]\n"
        return header + chunk
    except ImportError:
        return "(web_fetch requires httpx: pip install httpx)"
    except Exception as exc:
        return f"(fetch error: {exc})"


def _web_fetch_verify(
    emit_input: Dict, emit_output: str, verify_args: Dict
) -> Tuple[str, bool]:
    query = verify_args.get("query", emit_input.get("url", ""))
    if not query:
        return ("(no query — Frobenius trivially closed)", True)
    words = [w.lower() for w in query.split() if len(w) > 4]
    if not words:
        return ("(query too short for content check)", True)
    content_lower = emit_output.lower()
    matched = [w for w in words if w in content_lower]
    ratio   = len(matched) / len(words)
    if ratio >= 0.5:
        return (f"content relevance: {len(matched)}/{len(words)} query terms present ({ratio:.0%})", True)
    return (f"content may not address query: {len(matched)}/{len(words)} terms present ({ratio:.0%})", False)


# ── done ───────────────────────────────────────────────────────────────────────

def _done_emit(args: Dict[str, Any]) -> str:
    return args.get("conclusion", "(no conclusion provided)")


def _done_verify(
    emit_input: Dict, emit_output: str, verify_args: Dict
) -> Tuple[str, bool]:
    return ("(terminal action — Frobenius trivially closed)", True)


# ── Default dispatch tables ────────────────────────────────────────────────────

DEFAULT_EMIT_FNS: Dict[str, Any] = {
    "run_command":   _run_command_emit,
    "file_read":     _file_read_emit,
    "file_write":    _file_write_emit,
    "chunked_write": _chunked_write_emit,
    "web_fetch":     _web_fetch_emit,
    "done":          _done_emit,
}

DEFAULT_VERIFY_FNS: Dict[str, Any] = {
    "run_command":   _run_command_verify,
    "file_read":     _file_read_verify,
    "file_write":    _file_write_verify,
    "chunked_write": _chunked_write_verify,
    "web_fetch":     _web_fetch_verify,
    "done":          _done_verify,
}


# ── Tool schemas (OpenAI function-calling format) ──────────────────────────────

DEFAULT_TOOL_SCHEMAS: List[Dict] = [
    _fn(
        "run_command",
        (
            "Execute a shell command and return its stdout+stderr. "
            "Set `assertion` to a Python expression over `output` that must be True "
            "for Frobenius closure — e.g. '\"SUCCESS\" in output' or 'output.strip() != \"\"'. "
            "Leave `assertion` empty only when any output is acceptable."
        ),
        {
            "command":   _str("Shell command to run"),
            "assertion": _str("Python expression over `output` that must be True (optional)", ""),
            "timeout":   _int("Timeout in seconds", 30),
        },
        required=["command"],
    ),
    _fn(
        "file_read",
        (
            "Read a file from disk. Returns up to `limit` lines starting at `offset`. "
            "For large files, call repeatedly with increasing offsets."
        ),
        {
            "path":   _str("Absolute or relative path to the file"),
            "offset": _int("First line to return (0-indexed)", 0),
            "limit":  _int("Maximum number of lines to return", 200),
        },
        required=["path"],
    ),
    _fn(
        "file_write",
        (
            "Write content to a file (creates or overwrites). "
            "For files larger than ~4 KB, use chunked_write instead."
        ),
        {
            "path":    _str("Destination file path"),
            "content": _str("Full file content to write"),
        },
        required=["path", "content"],
    ),
    _fn(
        "chunked_write",
        (
            "Append or write a chunk to a file. Use mode='w' for the first chunk "
            "(creates/overwrites), mode='a' for subsequent chunks. "
            "Split content into ~3 KB pieces. Verification confirms bytes on disk."
        ),
        {
            "path":  _str("Destination file path"),
            "chunk": _str("Text chunk to write"),
            "mode":  _str("'w' to create/overwrite, 'a' to append", "a"),
        },
        required=["path", "chunk"],
    ),
    _fn(
        "web_fetch",
        (
            "Fetch a URL and return its text content. For large pages, use "
            "start_index and max_chars to paginate."
        ),
        {
            "url":         _str("URL to fetch"),
            "query":       _str("Optional search terms to verify content relevance", ""),
            "start_index": _int("Character offset to start reading from", 0),
            "max_chars":   _int("Maximum characters to return", 8000),
        },
        required=["url"],
    ),
    _fn(
        "done",
        "Signal that the task is complete. Write a full conclusion — this is the final output.",
        {"conclusion": _str("Complete answer or result for the task")},
        required=["conclusion"],
    ),
]
