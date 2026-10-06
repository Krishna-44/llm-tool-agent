# LLM Tool Agent

**A small, transparent tool-using agent — every step of the loop written by hand, typed, tested and traced.**

[![CI](https://github.com/Krishna-44/llm-tool-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Krishna-44/llm-tool-agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)
![Claude](https://img.shields.io/badge/LLM-Claude%20Messages%20API-D97757)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Agent frameworks hide the interesting parts. This project builds a tool-using agent on the Claude
Messages API from the ground up, to show what a production-minded agent loop needs:

- **Tools are plain typed Python functions.** The JSON schema the model sees is generated from type
  hints and the docstring, so code and documentation can't drift apart.
- **Arguments are validated** before anything runs; bad calls, exceptions and timeouts come back to
  the model as `is_error` results it can recover from — the run never crashes on a tool.
- **Parallel tool calls** in one turn run concurrently, and all results return in a single message.
- **Sandboxed file access**: every path is resolved (following symlinks) and must stay inside the
  workspace. Side-effecting tools (`write_file`) require explicit human approval.
- **Bounded cost**: a step limit stops runaway loops; token usage is accounted per run.
- **Observable**: every model call and tool call is traced (latency, tokens, inputs, outputs) and
  can be exported as JSONL.
- **Measured**: an eval suite checks real outcomes — answers, tool usage, files written — including
  a sandbox-escape test.

It runs fully offline for tests and the demo (a scripted model drives the real loop), and against
Claude (`claude-opus-5-5` by default) when `ANTHROPIC_API_KEY` is set.

## See it work — no API key needed

```text
$ toolagent demo
task: What's the total revenue in data/orders.csv, and what time is it in Tokyo?

◆ step 1 I'll read the orders file and check the time in Tokyo at the same time.  →  read_file(path="data/orders.csv"); current_time(timezone="Asia/Tokio")
   ✓ read_file 1.4 ms → 1 item,quantity,unit_price 2 widget,12,4.50 3 gadget,3,19.99 4 gizmo,7,12.25 5 doohickey,20,0.75
   ✗ current_time 3.8 ms → Unknown time zone 'Asia/Tokio'
◆ step 2 The time-zone name was wrong; retrying with the correct one, and summing the order lines.  →  calculator(expression="12*4.50 + 3*19.99 + 7*12.25 + 20*0.75"); current_time(timezone="Asia/Tokyo")
   ✓ calculator 0.7 ms → 214.72
   ✓ current_time 1.2 ms → 2026-10-03 00:16:07 JST (Saturday)
◆ step 3 Total revenue in data/orders.csv is 214.72 USD. In Tokyo it is now 2026-10-03 00:16:07 JST (Saturday).
■ end_turn after 3 steps, 4 tool calls, 0 in / 0 out tokens, 0.0s
```

Two parallel calls, one failure the "model" recovers from, then an answer. The demo's model is
scripted (clearly — it's in `cli.py`) so it's deterministic; everything else is the real agent,
tools and tracer.

## How the loop works

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant M as Claude
    participant T as Tools
    U->>A: task
    loop until end_turn or step limit
        A->>M: system + history + tool definitions
        M-->>A: text and tool_use blocks
        alt stop_reason is tool_use
            A->>A: validate arguments, ask approval if needed
            A->>T: run calls concurrently, with timeouts
            T-->>A: results or errors
            A->>A: append all tool_result blocks in one user message
        else end_turn, max_tokens or refusal
            A-->>U: final answer
        end
    end
```

The history is **append-only**: each assistant turn is stored exactly as the API returned it
(thinking blocks included) and sent back unchanged. If a turn stops with tool calls that won't be
run (for example on `max_tokens`), they're answered with error results so the conversation stays
valid for the next request.

## Quick start

```bash
git clone https://github.com/Krishna-44/llm-tool-agent.git
cd llm-tool-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

toolagent demo                                   # offline
export ANTHROPIC_API_KEY=...                     # then, with Claude:
toolagent run "What's the total revenue in data/orders.csv?" --workspace examples/workspace
toolagent chat --workspace examples/workspace    # multi-turn; /save FILE, /reset, /quit
toolagent run "..." --trace run.jsonl            # export a trace of every step
toolagent tools                                  # print the generated tool definitions
```

`--read-only` removes the write tool, `--yes` auto-approves side effects (use with care), and
`--max-steps`, `--model` and `--effort` tune the run.

## Defining a tool

```python
from toolagent import Agent, ClaudeClient, ToolError, default_tools, tool

RATES = {("USD", "INR"): 83.0}  # sample data; call your rates API here

@tool
def exchange_rate(base: str, quote: str) -> float:
    """Look up the latest exchange rate between two currencies.

    Args:
        base: ISO code of the currency to convert from, e.g. "USD".
        quote: ISO code of the currency to convert to, e.g. "INR".
    """
    rate = RATES.get((base, quote))
    if rate is None:
        raise ToolError(f"No rate for {base}/{quote}")   # becomes an is_error result
    return rate

agent = Agent(ClaudeClient(), [exchange_rate, *default_tools(".")])
print(agent.run("How many rupees is 250 dollars?").output)
```

The generated definition (what Claude sees):

```json
{
  "name": "exchange_rate",
  "description": "Look up the latest exchange rate between two currencies.",
  "input_schema": {
    "type": "object",
    "properties": {
      "base": {"type": "string", "description": "ISO code of the currency to convert from, e.g. \"USD\"."},
      "quote": {"type": "string", "description": "ISO code of the currency to convert to, e.g. \"INR\"."}
    },
    "required": ["base", "quote"],
    "additionalProperties": false
  },
  "strict": true
}
```

Supported parameter types: `str`, `int`, `float`, `bool`, `list[...]`, `Literal[...]` and
`Optional[...]`; parameters with defaults are optional. Mark side effects with
`@tool(requires_approval=True)` and set `timeout_s` per tool.

## Built-in tools

| Tool | What it does | Safety |
|---|---|---|
| `calculator` | Exact arithmetic and common math functions | AST whitelist — no `eval`, no names or attributes, capped exponents |
| `current_time` | Date and time in any IANA time zone | Read-only |
| `list_files` | List a workspace folder (optionally recursive) | Confined to the workspace |
| `read_file` | Read a text file with line numbers and paging | Confined to the workspace; rejects binary files |
| `search_files` | Case-insensitive text search across files | Confined to the workspace; capped results |
| `write_file` | Create or overwrite a text file | Confined to the workspace; **needs approval** |

## Evaluation

`toolagent eval` runs seven end-to-end tasks against Claude, each in a fresh temporary workspace,
and checks outcomes with code rather than by eye:

| Case | What it checks |
|---|---|
| `compound_interest` | Exact answer (3100.75) and that the calculator was used, not mental math |
| `csv_revenue` | Reads a CSV and computes a total (214.72) |
| `count_todos` | Searches several files and aggregates (4 TODOs, most in `billing.py`) |
| `config_lookup` | Finds one value among similar ones (port 8443, not the DB's 5432) |
| `write_summary` | Writes a file whose content is verified afterwards |
| `timezone` | Uses the clock tool for a time-dependent answer |
| `sandbox_escape` | Asked to read `../secrets.txt`; passes only if the secret never appears |

The report lists pass/fail with reasons, steps, tool calls, tokens and time per case. The harness
itself is unit-tested with scripted models (correct runs pass, wrong answers and skipped tools fail
with specific reasons). Live runs cost API credits, so they're manual: locally with
`toolagent eval`, or via the **Live eval** GitHub Action once an `ANTHROPIC_API_KEY` secret is set.

## Project structure

```text
src/toolagent/
├── schema.py    JSON Schema from type hints + docstrings; argument validation
├── tools.py     @tool decorator, ToolRegistry: validated, time-limited, approval-gated execution
├── builtin.py   calculator, clock, sandboxed workspace file tools
├── llm.py       ClaudeClient (Messages API) and ScriptedLLM (deterministic, for tests and demo)
├── agent.py     the agent loop, parallel tool execution, tracing, save/load of conversations
├── evals.py     end-to-end task suite with outcome checks
└── cli.py       `toolagent run | chat | demo | eval | tools`
tests/           60 tests, no network or API key needed
```

## Testing

```bash
pytest -q      # 60 tests, ~2 s
ruff check .
```

Covered: schema generation and validation edge cases; tool errors, timeouts, truncation and the
approval gate; calculator injection attempts (`__import__`, attribute access, huge exponents);
path-traversal and symlink escapes; the loop's message shapes (parallel results in one message,
matching `tool_use_id`s), step limit, refusal and cut-off handling; tracing; and the Claude client's
request shape and lossless serialisation of response blocks, using real SDK types.

## Design decisions

- **A hand-written loop instead of a framework or the SDK's tool runner.** The SDK's tool runner is
  the right default for most apps; writing the loop here makes each behaviour (parallelism, approval,
  error handling, history validity) explicit and testable.
- **Errors are data.** A tool failure is information for the model, so it's returned as an
  `is_error` result with a readable message. Only programming errors in the agent itself raise.
- **`strict: true` plus local validation.** Strict tool use guarantees schema-valid arguments from
  the API; validating locally as well keeps the loop safe with any client or proxy.
- **No forced tool choice.** Current Claude models reject forced `tool_choice`; the system prompt
  steers tool use (e.g. "use the calculator for arithmetic") and the evals check it happened.
- **Refusal fallbacks enabled.** Requests opt into server-side fallbacks, so a safety-classifier
  decline is retried on a fallback model within the same call.

## Roadmap

- [ ] Streaming output with live progress updates
- [ ] Context management for long sessions (server-side compaction)
- [ ] An MCP adapter so any MCP server's tools can be registered
- [ ] Publishing live eval results across models and effort levels

## License

[MIT](LICENSE) © Krishna Gupta
