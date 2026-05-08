"""
phi_c — The Phi_c boundary operator.

A self-verifying agentic loop where every action is paired with a verification step
that enforces the Frobenius condition: mu(delta(query)) == query.

Quick start:
    from phi_c import PhiCAgent
    agent = PhiCAgent(model="grok-4")
    result = agent.run_sync("Your task here")
"""

from .agent import PhiCAgent, LoopCycle, DualToolResult, LoopPhase
from .tools import DEFAULT_TOOL_SCHEMAS, DEFAULT_EMIT_FNS, DEFAULT_VERIFY_FNS

__all__ = [
    "PhiCAgent",
    "LoopCycle",
    "DualToolResult",
    "LoopPhase",
    "DEFAULT_TOOL_SCHEMAS",
    "DEFAULT_EMIT_FNS",
    "DEFAULT_VERIFY_FNS",
]
__version__ = "0.1.0"
