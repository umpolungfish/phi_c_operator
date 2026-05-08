"""
hello_world.py — Minimal phi-c usage.

Run:
    export OPENROUTER_API_KEY=your_key
    python examples/hello_world.py
"""

from phi_c import PhiCAgent

agent = PhiCAgent(model="grok-4")
result = agent.run_sync(
    "Run the shell command `echo 'hello from phi_c'` and report what it printed."
)
print(f"\nResult: {result}")
print(f"Frobenius ratio: {agent.frobenius_ratio:.2%}")
print(f"Windings: {len(agent.trajectory)}")
