"""A tiny SSE client. Run the server first, then: python client_demo.py

It asks one question, prints tokens as they arrive, then asks a follow-up on the
same thread. The follow-up only makes sense if the checkpointer remembered the
first answer.
"""

import json

import httpx

BASE_URL = "http://127.0.0.1:8000"


def stream_chat(message: str, thread_id: str | None = None):
    """Yield (event, data) pairs from the /chat/stream endpoint."""
    payload = {"message": message, "thread_id": thread_id}
    with httpx.stream("POST", f"{BASE_URL}/chat/stream", json=payload, timeout=60) as response:
        response.raise_for_status()
        event = None
        for line in response.iter_lines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                yield event, json.loads(line.removeprefix("data:"))
            # A blank line ends an event, but we already handled it at its data line.


def ask(message: str, thread_id: str | None = None) -> str | None:
    print(f"\nYou: {message}\nAssistant: ", end="")
    for event, data in stream_chat(message, thread_id):
        if event == "start":
            thread_id = data["thread_id"]
        elif event == "token":
            print(data["text"], end="", flush=True)
        elif event == "tool_call":
            print(f"\n  [calling {data['name']} ...]")
        elif event == "tool_result":
            print(f"  [{data['name']} returned {data['content']}]")
        elif event == "error":
            print(f"\n  [error: {data['detail']}]")
    print()
    return thread_id


if __name__ == "__main__":
    thread = ask("What is 1234 * 5678?")
    ask("Now divide that result by 2.", thread)  # "that result" needs the saved history
