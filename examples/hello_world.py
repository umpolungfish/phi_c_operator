"""
hello_world.py — Minimal odot usage.

Run:
    export OPENROUTER_API_KEY=your_key
    python examples/hello_world.py
"""

from odot import OdotAgent

agent = OdotAgent(model="grok-4")
result = agent.run_sync(
    "Run the shell command `echo 'hello from odot'` and report what it printed."
)
print(f"\nResult: {result}")
print(f"Frobenius ratio: {agent.frobenius_ratio:.2%}")
print(f"Windings: {len(agent.trajectory)}")
