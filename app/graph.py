"""The agent itself: a small LangGraph state machine.

    START --> agent --(model asked for tools)--> tools --+
                ^                                        |
                +----------------------------------------+
                |
                +--(model gave a final answer)--> END

Three LangGraph ideas to look for below:

1. STATE. A shared object that every node reads and updates. Here it is
   MessagesState, which is just {"messages": [...]}. A node returns only the
   *changes* it wants to make, and a "reducer" merges them in. For messages the
   reducer (add_messages) appends, so the list grows into the whole conversation.

2. NODES and EDGES. Nodes are functions (state in, partial state out). Edges say
   which node runs next. A *conditional* edge chooses the next node by looking at
   the state, and that choice is what turns a straight line into an agent loop.

3. CHECKPOINTER. After every node finishes, LangGraph saves the state under a
   thread_id. Invoke the graph again with the same thread_id and it resumes from
   the saved state. That is the entire "conversation memory" mechanism.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.config import get_settings
from app.tools import TOOLS


def build_graph(checkpointer: BaseCheckpointSaver, llm: BaseChatModel | None = None):
    """Assemble and compile the agent graph.

    The model is a parameter so tests can hand in a fake one (see tests/test_app.py).
    In normal use it is None and we build a real ChatAnthropic.
    """
    settings = get_settings()

    if llm is None:
        llm = ChatAnthropic(model=settings.model_name, max_tokens=2048)

    # bind_tools attaches the tool schemas to every request so the model *can*
    # ask for a tool. It only advertises the tools; nothing runs here.
    model = llm.bind_tools(TOOLS)

    async def call_model(state: MessagesState, config: RunnableConfig) -> dict:
        # The system prompt is added here, at call time, instead of being stored
        # in state. It is not part of the conversation, so it stays out of saved
        # history and you can edit it without migrating existing threads.
        messages = [SystemMessage(settings.system_prompt), *state["messages"]]

        # Passing `config` along lets LangGraph's streaming callbacks follow the
        # model call. It is required on Python < 3.11 and harmless on newer ones.
        response = await model.ainvoke(messages, config)

        # Return only the change: one new message. add_messages appends it.
        return {"messages": [response]}

    builder = StateGraph(MessagesState)

    builder.add_node("agent", call_model)
    # ToolNode runs every tool call found in the last AI message and appends one
    # ToolMessage per call with the result.
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "agent")
    # tools_condition is a ready-made router: if the last AI message contains
    # tool calls it returns "tools", otherwise END.
    builder.add_conditional_edges("agent", tools_condition)
    # After tools run, always go back to the model so it can read the results.
    builder.add_edge("tools", "agent")

    # The checkpointer is what makes thread_id-based memory work.
    return builder.compile(checkpointer=checkpointer)
