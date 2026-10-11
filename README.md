# LangGraph + FastAPI agent

A small chat agent you can learn from by reading the code. Claude (through
LangGraph) can call tools, replies stream to the client token by token, and
conversations are remembered between requests.

## Reading order

Each file does one job and is commented to explain the *why*. Read them in this order:

| # | File | What you learn |
|---|------|----------------|
| 1 | `app/tools.py` | How a Python function becomes a tool the model can call |
| 2 | `app/graph.py` | **The core.** State, nodes, conditional edges, the agent loop, the checkpointer |
| 3 | `app/streaming.py` | How LangGraph's `stream_mode="messages"` becomes clean events |
| 4 | `app/main.py` | FastAPI wiring: lifespan, dependencies, SSE, WebSocket, history |
| 5 | `tests/test_app.py` | How to test an agent offline with a scripted fake model |
| - | `app/config.py`, `client_demo.py` | Settings, and a tiny SSE client |

## How a request flows

```
client ──POST /chat/stream {message, thread_id}──▶ main.py
                                                     │
                                          streaming.py: graph.astream(...)
                                                     │
                              ┌──────────────────────▼──────────────────────┐
                              │ graph.py                                    │
                              │   START → agent ──tool calls?──▶ tools ─┐   │
                              │             ▲                           │   │
                              │             └───────────────────────────┘   │
                              │   (checkpointer saves state after each node)│
                              └──────────────────────┬──────────────────────┘
                                                     │ token / tool_call / tool_result
client ◀──────── SSE events: start, token…, done ────┘
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then put your real ANTHROPIC_API_KEY in .env
uvicorn app.main:app --reload
```

Interactive API docs: http://127.0.0.1:8000/docs

## Try it

```bash
# Streaming (SSE). -N stops curl from buffering. The first event contains your thread_id.
curl -N localhost:8000/chat/stream -H 'content-type: application/json' \
  -d '{"message": "What is 1234 * 5678?"}'

# Continue the same conversation by sending the thread_id back
curl -N localhost:8000/chat/stream -H 'content-type: application/json' \
  -d '{"message": "Now divide that by 2", "thread_id": "PASTE_THREAD_ID"}'

# One-shot, no streaming
curl localhost:8000/chat -H 'content-type: application/json' -d '{"message": "What time is it in Lagos?"}'

# Read a conversation back out of the checkpointer, or delete it
curl localhost:8000/threads/PASTE_THREAD_ID/messages
curl -X DELETE localhost:8000/threads/PASTE_THREAD_ID
```

Or run the demo client, which streams a question and a follow-up: `python client_demo.py`

WebSocket (the connection stays open for many turns):

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8000/ws/chat") as ws:
        print(await ws.recv())  # {"event": "start", "thread_id": ...}
        await ws.send(json.dumps({"message": "What is 2 ** 10?"}))
        while True:
            event = json.loads(await ws.recv())
            print(event)
            if event["event"] in ("done", "error"):
                break

asyncio.run(main())
```

## Run the tests

```bash
pytest -q
```

No API key or network needed. A scripted fake model replaces Claude, while the
real graph, checkpointer and endpoints run.

## Where memory lives

Conversation state is saved to `checkpoints.sqlite` (tables `checkpoints` and
`writes`), keyed by `thread_id`. Restart the server and old threads still work.
Open the file in any SQLite viewer to see what LangGraph stores.

## Things to try next

- **Add a tool** in `tools.py` (append it to `TOOLS`) and ask the model to use it.
- **Change the model** with `MODEL_NAME` in `.env`, or swap `ChatAnthropic` in `graph.py` for another LangChain chat model.
- **Print the state** after each step: change `stream_mode` in `streaming.py` to `"updates"` and watch what each node returns.
- **Add a human approval step**: look up LangGraph's `interrupt()`. The checkpointer is what makes pausing and resuming possible.
- **Count tokens or trim history**: long threads grow without bound. Look at `trim_messages` and apply it inside `call_model`.

## Known limits (deliberate, to keep the code readable)

- **No authentication.** Anyone who can reach the server can read any thread whose id they know. Add auth and tie `thread_id` to a user before exposing it.
- **One request at a time per thread.** Two simultaneous requests on the same `thread_id` can interleave their messages.
- **SQLite suits a single process.** For several workers or servers, switch to the Postgres checkpointer (`langgraph-checkpoint-postgres`).
- **Raw error text is sent to the client** in `error` events. Replace it with a generic message in production.
