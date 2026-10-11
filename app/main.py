"""The HTTP layer: exposes the graph through FastAPI.

    POST   /chat                       one-shot: wait for the full reply
    POST   /chat/stream                Server-Sent Events (SSE) stream
    WS     /ws/chat                    WebSocket, many turns on one connection
    GET    /threads/{id}/messages      read a conversation back from the checkpointer
    DELETE /threads/{id}               forget a conversation
    GET    /health

Every chat endpoint takes an optional thread_id. Omit it to start a new
conversation (the server invents one and tells you); send it back on later
requests to continue the same conversation.
"""

import json
import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field
from starlette.requests import HTTPConnection

from app.config import get_settings
from app.graph import build_graph
from app.streaming import stream_agent, text_of, thread_config

router = APIRouter()


# --- request model and dependencies --------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = None  # omit to start a new conversation


# The graph and checkpointer are built once at startup (see create_app) and
# parked on app.state. These two small functions let endpoints ask for them with
# Depends(...). HTTPConnection is the base class of both Request and WebSocket,
# so the same dependency works for HTTP and WebSocket routes.
def get_graph(conn: HTTPConnection):
    return conn.app.state.graph


def get_checkpointer(conn: HTTPConnection):
    return conn.app.state.checkpointer


def new_thread_id() -> str:
    return uuid.uuid4().hex


# --- endpoints -------------------------------------------------------------------


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/chat")
async def chat(req: ChatRequest, graph=Depends(get_graph)):
    """The simplest possible use of the graph: ainvoke runs it to completion."""
    thread_id = req.thread_id or new_thread_id()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=req.message)]}, thread_config(thread_id)
    )
    # The final state's last message is the assistant's answer.
    return {"thread_id": thread_id, "reply": text_of(result["messages"][-1].content)}


def sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event. The wire format is plain text: an `event:`
    line, a `data:` line, then a blank line that marks the end of the event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, graph=Depends(get_graph)):
    """SSE streaming.

    Note: the browser's built-in EventSource only supports GET requests. From a
    browser, call this with fetch() and read response.body as a stream, or switch
    the route to GET with the message in a query parameter.
    """
    thread_id = req.thread_id or new_thread_id()

    async def event_source():
        # The thread_id goes out first so a client that didn't send one learns it.
        yield sse("start", {"thread_id": thread_id})
        try:
            async for event, payload in stream_agent(graph, req.message, thread_id):
                yield sse(event, payload)
            yield sse("done", {})
        except Exception as exc:
            # By now the 200 status line has been sent, so we can no longer
            # return an HTTP error. The only way to report failure is in-band.
            # (Client disconnects raise CancelledError, a BaseException, so they
            # pass through here untouched.) In production, don't echo str(exc).
            yield sse("error", {"detail": str(exc)})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # no proxy buffering
    )


@router.websocket("/ws/chat")
async def ws_chat(ws: WebSocket, thread_id: str | None = None, graph=Depends(get_graph)):
    """WebSocket streaming. Unlike SSE, the connection stays open across turns:
    the client sends {"message": "..."} as often as it likes, and each turn
    streams back JSON events ending with {"event": "done"}."""
    await ws.accept()
    thread_id = thread_id or new_thread_id()
    await ws.send_json({"event": "start", "thread_id": thread_id})
    try:
        while True:
            data = await ws.receive_json()
            message = str(data.get("message", "")).strip()
            if not message:
                await ws.send_json({"event": "error", "detail": "Send {'message': '...'}"})
                continue
            try:
                async for event, payload in stream_agent(graph, message, thread_id):
                    await ws.send_json({"event": event, **payload})
                await ws.send_json({"event": "done"})
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                await ws.send_json({"event": "error", "detail": str(exc)})
    except WebSocketDisconnect:
        pass  # the client closed the connection; nothing to clean up


def serialize(message) -> dict:
    """Shape a LangChain message into plain JSON for the history endpoint."""
    role = {"human": "user", "ai": "assistant", "tool": "tool"}.get(message.type, message.type)
    item = {"role": role, "content": text_of(message.content)}
    if isinstance(message, AIMessage) and message.tool_calls:
        item["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in message.tool_calls]
    return item


@router.get("/threads/{thread_id}/messages")
async def get_messages(thread_id: str, graph=Depends(get_graph)):
    """Read a conversation straight out of the checkpointer."""
    snapshot = await graph.aget_state(thread_config(thread_id))
    messages = snapshot.values.get("messages", [])
    if not messages:
        raise HTTPException(status_code=404, detail="Unknown thread")
    return {"thread_id": thread_id, "messages": [serialize(m) for m in messages]}


@router.delete("/threads/{thread_id}", status_code=204)
async def delete_thread(thread_id: str, checkpointer=Depends(get_checkpointer)):
    await checkpointer.adelete_thread(thread_id)


# --- application factory -----------------------------------------------------------


def create_app(llm: BaseChatModel | None = None, checkpoint_db: str | None = None) -> FastAPI:
    """Build the FastAPI app. The two parameters exist for tests: pass a fake
    model and ":memory:" to run everything offline. Normal use passes neither."""
    db_path = checkpoint_db or get_settings().checkpoint_db

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Runs once at startup (before `yield`) and once at shutdown (after).
        # `async with` opens the SQLite connection now and closes it cleanly later.
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
            app.state.checkpointer = checkpointer
            app.state.graph = build_graph(checkpointer, llm)
            yield

    app = FastAPI(title="LangGraph agent", lifespan=lifespan)
    app.include_router(router)
    return app


# `uvicorn app.main:app` looks for this name.
app = create_app()
