# agents-101 — Project Knowledge

## מטרה
מחברת Jupyter חינוכית שמסבירה את הקשר בין LangChain, LangGraph ו-MCP לבניית סוכני AI.

## קבצים
- `agents_101.ipynb` — המחברת הראשית
- `requirements.txt` — תלויות Python

## מבנה המחברת
1. **מבוא** — מה זה סוכן AI ולמה צריך frameworks
2. **חלק א: LangChain** — ChatModel, Tools, bind_tools
3. **חלק ב: LangGraph** — StateGraph, nodes, edges, checkpointing, human-in-the-loop
4. **חלק ג: MCP** — הבעיה שפותרת, MCP Server (FastMCP), langchain-mcp-adapters
5. **חלק ד: Big Picture** — איך כל השכבות מתחברות
6. **בונוס** — Human-in-the-Loop עם interrupt

## Stack
- Python 3.10+
- LangChain + langchain-anthropic
- LangGraph
- MCP + langchain-mcp-adapters
- Claude Sonnet 4.6 (claude-sonnet-4-6)

## הערות
- דורש ANTHROPIC_API_KEY בסביבה
- המחברת מדגימה stdio transport ל-MCP (הפשוט ביותר)
