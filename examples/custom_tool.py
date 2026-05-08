"""
custom_tool.py — Registering a custom tool with emit + verify.

Demonstrates a simple database query tool whose verify step
checks that the result is non-empty JSON.

Run:
    export OPENROUTER_API_KEY=your_key
    python examples/custom_tool.py
"""

import json
from phi_c import PhiCAgent

# Emit: perform the action, return a string
def db_query_emit(args):
    sql = args.get("sql", "")
    # Simulated response — replace with a real DB call
    fake_data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    return json.dumps(fake_data)

# Verify: confirm the result satisfies the Frobenius condition
def db_query_verify(emit_input, emit_output, verify_args):
    try:
        data = json.loads(emit_output)
        if isinstance(data, list) and len(data) > 0:
            return (f"query returned {len(data)} row(s)", True)
        return ("query returned empty result", False)
    except json.JSONDecodeError:
        return ("result is not valid JSON", False)

db_query_schema = {
    "type": "function",
    "function": {
        "name": "db_query",
        "description": "Run a SQL query and return results as JSON.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL query to execute"},
            },
            "required": ["sql"],
        },
    },
}

agent = PhiCAgent(model="grok-4")
agent.register_tool("db_query", db_query_schema, db_query_emit, db_query_verify)

result = agent.run_sync("Query the users table (SELECT * FROM users) and report how many users exist.")
print(f"\nResult: {result}")
print(f"\nStructural type: {json.dumps(agent.structural_type, indent=2)}")
