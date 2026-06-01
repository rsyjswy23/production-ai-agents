# Production AI Agents

Multi-agent AI systems built with LangChain, LangGraph, and FastAPI, featuring RAG pipelines, vector databases, LLM orchestration, secure prompt handling, evaluation workflows, and production deployment.

Highlights
- Key features: Chains & LCEL, prompt composition, structured outputs (Pydantic), LangSmith tracing, token-cost analysis, multi-model support, and debug tooling.

Quickstart
- Prereqs: set API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `LANGSMITH_API_KEY`) in a `.env` file or your shell.
- Install dependencies (see `pyproject.toml`) — example using pip:
```bash
pip install -r requirements.txt
```
- Run the Smart Q&A bot demo:
```bash
python smart_bot_section1.py
```

Environment notes
- Demos call `load_dotenv()`; secrets may be provided via a `.env` file.
- To enable LangSmith tracing, set `LANGSMITH_API_KEY`. Optionally set `LANGSMITH_PROJECT` to group traces.

Files (short overview)
- `smart_bot_section1.py` — Smart Q&A bot with `QAResponse` schema, `SmartQABot` class, and LangSmith `@traceable` usage (batch/error demos).
- `chains_v1.py` — LCEL chain patterns: basic chains, `RunnableParallel`, passthroughs, `RunnableBranch`, and debugging examples.
- `core_concepts.py` — Core LCEL/runnable demos, streaming, schema inspection, and exercises.
- `prompt_templates_all.py` — Prompt templates, placeholders, few-shot examples, and composition.
- `prompt_messages.py` — Message templates and few-shot prompt construction.
- `working_with_llms.py` — Multi-provider LLM initialization, model comparisons, streaming, and cost-aware patterns.
- `output_parsers_demo.py` — `StrOutputParser`, `JsonOutputParser`, and Pydantic/structured examples.
- `output_parsers_final.py` — Detailed structured-output demos and complex schemas.
- `main.py` — Quick provider/version checks and smoke tests.

LangSmith (traces & visuals)
- Smart Q&A demo screenshot:  
	![Smart Q&A Demo](img/demo-bot.jpg)
- LangSmith run trace example:  
	![LangSmith Tracing](img/LangSmith-tracing.jpg)
- Token cost breakdown per query:  
	![LangSmith Token Cost](img/LangSmith-tokencost.jpg)

Usage tips
- Prefer `with_structured_output()` or `PydanticOutputParser` for type-safe structured results.
- Use `RunnableParallel` for concurrent subtasks and `RunnableBranch` for intent-based routing.
- Use LangSmith to identify bottlenecks and token-cost hotspots; enable tracing with `LANGSMITH_API_KEY`.

