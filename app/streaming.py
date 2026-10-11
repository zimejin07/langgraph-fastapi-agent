"""Turn LangGraph's raw stream into small, client-friendly events.

The SSE and WebSocket endpoints both use stream_agent(), so everything
LangGraph-specific about streaming lives in this one file.

Events are yielded as (event_name, payload) tuples:

    token        {"text": "Hel"}                    a piece of the assistant's reply
    tool_call    {"name": "calculate"}              the model decided to call a tool
    tool_result  {"name": "calculate", "content": "4"}   the tool finished
"""

from collections.abc import AsyncIterator

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage


def text_of(content: str | list) -> str:
    """Message content is a plain string for most models, but Anthropic can return
    a list of content blocks like [{"type": "text", "text": "..."}]. Flatten both."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def thread_config(thread_id: str) -> dict:
    """The `configurable` dict is how you tell the checkpointer which conversation this is."""
    return {"configurable": {"thread_id": thread_id}}


async def stream_agent(graph, message: str, thread_id: str) -> AsyncIterator[tuple[str, dict]]:
    # We send only the NEW user message. The checkpointer loads the earlier
    # messages for this thread_id and add_messages appends ours to them.
    inputs = {"messages": [HumanMessage(content=message)]}

    # LangGraph has several stream modes:
    #   "values"   full state after every step
    #   "updates"  just what each node returned
    #   "messages" LLM output token by token  <- the one we want for a chat UI
    # In "messages" mode each item is a (message_chunk, metadata) tuple, and
    # metadata["langgraph_node"] names the graph node that produced it.
    async for msg, meta in graph.astream(inputs, thread_config(thread_id), stream_mode="messages"):
        node = meta["langgraph_node"]

        # Filtering on node name AND message type means only fresh output from
        # this run reaches the client, never the user's own message or old history.
        if node == "agent" and isinstance(msg, AIMessageChunk):
            if text := text_of(msg.content):
                yield "token", {"text": text}
            # A tool call arrives in pieces: the first piece carries the tool's
            # name and later pieces carry fragments of the JSON arguments. We
            # announce the call when we see the name and skip the fragments.
            for call in msg.tool_call_chunks:
                if call.get("name"):
                    yield "tool_call", {"name": call["name"]}

        elif node == "tools" and isinstance(msg, ToolMessage):
            yield "tool_result", {"name": msg.name, "content": text_of(msg.content)}
