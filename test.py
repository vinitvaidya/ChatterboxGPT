from agent import get_agent
from langchain_core.messages import HumanMessage, AIMessageChunk
from database import init_db

init_db()

agent = get_agent("gemini-3.8-flash")
config = {"configurable": {"thread_id": "test_thread_id"}}


def extract_text(content) -> str:
    """Return only the text from a str or a list of content blocks."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


for chunk, metadata in agent.stream(
    {"messages": [HumanMessage(content="Generate a blog about Machine Learning")]},
    config=config,
    stream_mode="messages",
):
    if isinstance(chunk, AIMessageChunk):
        text = extract_text(chunk.content)
        if text:
            print(text, end="", flush=True)
