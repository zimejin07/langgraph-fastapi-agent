"""Offline tests: no API key, no network.

A scripted fake chat model stands in for Claude, so these tests exercise the REAL
graph, the REAL checkpointer and the REAL FastAPI endpoints. Only the model is
fake. Read them to see exactly what each endpoint emits.

Run with:  pytest -q
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from app.main import create_app


class ScriptedChatModel(BaseChatModel):
    """Replays pre-written AIMessages, one per model call, and records how many
    messages it was shown each time (which is how we prove memory works)."""

    script: list[AIMessage]
    seen_message_counts: list[int] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self  # the script already decides when tools get called

    def _next_reply(self, messages) -> AIMessage:
        self.seen_message_counts.append(len(messages))
        return self.script.pop(0)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next_reply(messages))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        # Mimic real streaming: tool calls arrive as chunks, text arrives word by word.
        reply = self._next_reply(messages)
        if reply.tool_calls:
            for index, call in enumerate(reply.tool_calls):
                chunk = AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": call["name"],
                            "args": json.dumps(call["args"]),
                            "id": call["id"],
                            "index": index,
                        }
                    ],
                )
                yield ChatGenerationChunk(message=chunk)
        else:
            for word in reply.content.split(" "):
                yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))


def calc_call(expression: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "calculate", "args": {"expression": expression}, "id": call_id}],
    )


@pytest.fixture
def client():
    # Two full turns: each is "call the calculator", then "give the answer".
    model = ScriptedChatModel(
        script=[
            calc_call("2 + 2", "call_1"),
            AIMessage(content="It is 4."),
            calc_call("4 * 10", "call_2"),
            AIMessage(content="That is 40."),
        ]
    )
    with TestClient(create_app(llm=model, checkpoint_db=":memory:")) as test_client:
        test_client.model = model
        yield test_client


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_sse_streams_tokens_and_tool_events(client):
    events = parse_sse(client.post("/chat/stream", json={"message": "What is 2+2?"}).text)
    names = [name for name, _ in events]

    assert names[0] == "start" and names[-1] == "done"
    assert names.count("tool_call") == 1
    assert names.count("tool_result") == 1
    assert ("tool_call", {"name": "calculate"}) in events
    assert ("tool_result", {"name": "calculate", "content": "4"}) in events
    # The reply arrives as several small tokens, not one big blob.
    tokens = [data["text"] for name, data in events if name == "token"]
    assert len(tokens) > 1
    assert "".join(tokens).strip() == "It is 4."


def test_checkpointer_gives_the_model_its_history(client):
    first = parse_sse(client.post("/chat/stream", json={"message": "What is 2+2?"}).text)
    thread_id = first[0][1]["thread_id"]

    second = parse_sse(
        client.post("/chat/stream", json={"message": "Times ten?", "thread_id": thread_id}).text
    )

    # Only the NEW turn's tool result is streamed (no replay of old ones).
    results = [data for name, data in second if name == "tool_result"]
    assert results == [{"name": "calculate", "content": "40"}]

    # The model saw a longer and longer message list: system + human = 2, then
    # + AI tool call + tool result = 4, then turn two starts from the 4 saved
    # messages plus the new question = 6, and so on. That growth IS the memory.
    assert client.model.seen_message_counts == [2, 4, 6, 8]


def test_history_endpoint_and_delete(client):
    thread_id = parse_sse(client.post("/chat/stream", json={"message": "What is 2+2?"}).text)[0][1][
        "thread_id"
    ]

    history = client.get(f"/threads/{thread_id}/messages").json()["messages"]
    assert [m["role"] for m in history] == ["user", "assistant", "tool", "assistant"]
    assert history[1]["tool_calls"] == [{"name": "calculate", "args": {"expression": "2 + 2"}}]
    assert history[-1]["content"].strip() == "It is 4."

    assert client.delete(f"/threads/{thread_id}").status_code == 204
    assert client.get(f"/threads/{thread_id}/messages").status_code == 404


def test_unknown_thread_is_404(client):
    assert client.get("/threads/nope/messages").status_code == 404


def test_non_streaming_chat(client):
    body = client.post("/chat", json={"message": "What is 2+2?"}).json()
    assert body["reply"].strip() == "It is 4."
    assert body["thread_id"]


def test_websocket_multiple_turns_on_one_connection(client):
    with client.websocket_connect("/ws/chat") as ws:
        assert ws.receive_json()["event"] == "start"

        for question, expected in [("What is 2+2?", "It is 4."), ("Times ten?", "That is 40.")]:
            ws.send_json({"message": question})
            events = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["event"] == "done":
                    break
            text = "".join(e["text"] for e in events if e["event"] == "token")
            assert text.strip() == expected
            assert any(e["event"] == "tool_result" for e in events)


def test_empty_message_is_rejected(client):
    assert client.post("/chat/stream", json={"message": ""}).status_code == 422


def test_real_model_class_builds_without_network(monkeypatch, tmp_path):
    # Constructing ChatAnthropic makes no network call, so we can at least prove
    # the production wiring starts up. (We can't test real replies offline.)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with TestClient(create_app(checkpoint_db=str(tmp_path / "db.sqlite"))) as test_client:
        assert test_client.get("/health").status_code == 200
