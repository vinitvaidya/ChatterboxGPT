import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
import certifi  # to prevent windows path related errors

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from datetime import date
from tools import tools

Path("data").mkdir(exist_ok=True)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

ALLOWED_MODELS = {
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.8-flash-lite"
    "us.meta.llama4-scout-17b-instruct-v1:0",
    "us.zai.glm-5",
}


SYSTEM_PROMPT = f"""
You are ChatterboxGPT, a helpful AI assistant with access to tools. Today's date is {date.today().isoformat()}.

## Tool usage
- search_web (Tavily): Use for anything time-sensitive or likely to have changed: news, current events, prices, recent releases, current office holders, latest software versions. Summarize the findings, note that they come from web search, and cite source URLs.
- search_uploaded_documents: Use whenever the user refers to an uploaded file or document. If it returns nothing relevant, say so; do not guess at the contents.
- remember_this: Use when the user asks you to remember something, or when they share a stable preference or fact that will clearly be useful later (name, job, preferences). Never store passwords, financial details, or other sensitive data.
- recall_memory: Use when the user refers to past preferences or saved facts, or when personalization would clearly improve your answer.
- calculator: Use for non-trivial calculations. Simple arithmetic you can do directly.

## Guidelines
- Answer directly without tools when you already know the answer and it isn't time-sensitive.
- Never fabricate tool outputs, sources, or document contents. If tools return nothing useful, say so.
- If a request is ambiguous, ask one short clarifying question.
- Be clear, accurate, and concise.
"""


def normalize_model_name(model_name: str | None) -> str:
    """
    Validate the selected model from frontend.
    If model is missing or not allowed, fallback to DEFAULT_MODEL.
    """
    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name


def build_agent(model_name: str):
    """
    Build Langgraph agent for a selected Gemini model.
    """
    selected_model = normalize_model_name(model_name)

    if "gemini" in selected_model:
        llm = ChatGoogleGenerativeAI(model=selected_model, temperature=0.3)

    else:
        llm = ChatBedrockConverse(
            model=selected_model,  # inference profile ID or model ID
            region_name="us-east-1",
            temperature=0.3,
        )

    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)

        return {"messages": [response]}

    tool_node = ToolNode(tools)

    workflow = StateGraph(MessagesState)

    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools", "chatbot")

    conn = sqlite3.connect("data/langgraph_checkpoints.sqlite", check_same_thread=False)

    checkpointer = SqliteSaver(conn)

    return workflow.compile(checkpointer=checkpointer)


_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):
    """
    Return cached LangGraph agent for selected model.
    If not created yet, create it once and reuse it.
    """

    selected_model = normalize_model_name(model_name)

    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)

    return _AGENT_CACHE[selected_model]
