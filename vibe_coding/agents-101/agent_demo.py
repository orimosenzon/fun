#!/usr/bin/env python3
"""
agent_demo.py — סוכן LangGraph מינימלי עם חישוב ואינטרנט
=====================================================

מבנה:
  1. שני כלים: calculate (חישוב בטוח) + web_search (DuckDuckGo)
  2. גרף LangGraph: START → agent ⇄ tools → END
  3. זיכרון קבוע בין הרצות — SQLite (agent_memory.db)

להרצה:
  pip install duckduckgo-search langgraph-checkpoint-sqlite
  echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
  python agent_demo.py
"""

import os
import ast
import operator
from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from ddgs import DDGS


# ─────────────────────────────────────────
# מפתח API — נטען מקובץ .env
# ─────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
if not os.environ.get("ANTHROPIC_API_KEY"):
    raise RuntimeError("חסר ANTHROPIC_API_KEY — צור קובץ .env עם השורה:\nANTHROPIC_API_KEY=sk-ant-...")


# ─────────────────────────────────────────
# כלי 1: חישוב אריתמטי בטוח
#   לא משתמשים ב-eval() הגולמי —
#   מפרקים לעץ AST ומריצים רק פעולות מתמטיות
# ─────────────────────────────────────────
_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.Mod:  operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _eval_ast(node):
    if isinstance(node, ast.Constant):
        return node.n
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval_ast(node.operand))
    raise ValueError(f"פעולה לא נתמכת: {type(node).__name__}")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    Examples: '2*200/10+5', '(3+4)**2', '100/4 + 7'"""
    try:
        result = _eval_ast(ast.parse(expression.strip(), mode="eval").body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"שגיאה בחישוב: {e}"


# ─────────────────────────────────────────
# כלי 2: חיפוש ברשת — DuckDuckGo (חינמי, ללא API key)
# ─────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """Search the web for current information, news, or facts.
    Use when asked about recent events or things you don't know."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "לא נמצאו תוצאות."
        return "\n\n".join(
            f"[{r['title']}]\n{r['body']}"
            for r in results
        )
    except Exception as e:
        return f"שגיאת חיפוש: {e}"


# ─────────────────────────────────────────
# הגרף
#
#  START
#    ↓
#  [agent]  ← קריאה ל-LLM, מחליט אם לקרוא לכלי
#    ↓  ↑
#  tools_condition:
#    אם LLM ביקש כלי → [tools] → חזרה ל-[agent]
#    אם לא           → END
# ─────────────────────────────────────────
tools = [calculate, web_search]

# claude-haiku — המודל הזול ביותר של Anthropic
model = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def agent_node(state: State):
    """צומת הסוכן: שולח את כל ההיסטוריה ל-LLM ומקבל תשובה / בקשת כלי."""
    return {"messages": [model.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # כלי? → tools; לא? → END
builder.add_edge("tools", "agent")                       # אחרי כלי — חזרה לסוכן


# ─────────────────────────────────────────
# זיכרון קבוע בין הרצות — SQLite
#   agent_memory.db נשמר לצד הסקריפט
#   thread_id="main" = שיחה אחת מתמשכת
# ─────────────────────────────────────────
MEMORY_DB = Path(__file__).parent / "agent_memory.db"
THREAD_ID = "main"
config = {"configurable": {"thread_id": THREAD_ID}}


# ─────────────────────────────────────────
# לולאת השיחה
# ─────────────────────────────────────────
def main():
    with SqliteSaver.from_conn_string(str(MEMORY_DB)) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        print("=" * 50)
        print(" סוכן מוכן לשימוש")
        print(f" זיכרון: {MEMORY_DB.name}")
        print(" כתוב 'יציאה' או 'exit' לעצירה")
        print("=" * 50)

        while True:
            try:
                user_input = input("\nאתה: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nלהתראות!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("יציאה", "exit", "quit"):
                print("להתראות!")
                break

            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )

            response = result["messages"][-1].content
            print(f"\nסוכן: {response}")


if __name__ == "__main__":
    main()
