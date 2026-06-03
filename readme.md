# LangGraph Chatbot Service

A production-style conversational AI backend built with **LangGraph** and **FastAPI**, supporting both cloud LLM APIs (OpenAI) and local model serving (Ollama). Designed as a stateful, multi-turn chat service with OpenAI-compatible REST API, streaming responses, and a Gradio web UI.

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Client Layer                      │
│         Gradio Web UI  /  REST API Clients          │
└──────────────────────┬──────────────────────────────┘
                       │  HTTP POST /v1/chat/completions
                       ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI Service (uvicorn)              │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │          LangGraph StateGraph               │    │
│  │                                             │    │
│  │   START → [chatbot node] → END              │    │
│  │              │                              │    │
│  │         MemorySaver                         │    │
│  │   (per userId + conversationId session)     │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  ┌──────────────┐    ┌─────────────────────────┐    │
│  │  Prompt Tmpl │    │      LLM Backend        │    │
│  │  system.txt  │    │  OpenAI gpt-4o-mini  OR │    │
│  │  user.txt    │    │  Ollama deepseek-r1:14b │    │
│  └──────────────┘    └─────────────────────────┘    │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
         LangSmith Tracing (optional)
```

---

## Key Features

**Stateful Multi-Turn Conversation**
LangGraph `MemorySaver` checkpoints conversation state per `userId + conversationId`, maintaining context across multiple requests without client-side history management.

**OpenAI-Compatible REST API**
`POST /v1/chat/completions` follows the OpenAI chat completion schema — drop-in compatible with any client built against the OpenAI API spec.

**Streaming & Non-Streaming Responses**
Supports both Server-Sent Events (SSE) streaming via `StreamingResponse + graph.astream()` and standard JSON responses, selectable per request via the `stream` flag.

**Switchable LLM Backend**
Single config change in `llms.py` (`DEFAULT_LLM_TYPE`) swaps between:
- **OpenAI** — `gpt-4o-mini` via `ChatOpenAI`
- **Ollama (local)** — `deepseek-r1:14b` via `ChatOllama` at `localhost:11434`

Fallback logic: if the requested model fails to initialize, automatically falls back to the default.

**Versioned Prompt Templates**
System and user prompts are externalized to `prompt_template_system.txt` and `prompt_template_user.txt` — editable without touching application code, friendly to prompt version control.

**LangSmith Observability**
Integrates `LANGCHAIN_TRACING_V2` for end-to-end request tracing, useful for evaluating response quality and debugging prompt regressions.

---

## Project Structure

```
langGraphChatbot/
├── main.py                    # FastAPI app, LangGraph graph, API endpoints
├── llms.py                    # LLM abstraction: OpenAI / Ollama initialization + fallback
├── webUI.py                   # Gradio web UI (runs on port 7860)
├── apiTest.py                 # REST API test client
├── prompt_template_system.txt # System prompt template
├── prompt_template_user.txt   # User prompt template (wraps user query)
├── graph.png                  # Auto-generated LangGraph visualization
├── requirements.txt
└── .env                       # API keys (not committed)
```

---

## API Reference

### `POST /v1/chat/completions`

Request body:

```json
{
  "messages": [
    {"role": "user", "content": "your question here"}
  ],
  "stream": false,
  "userId": "user_123",
  "conversationId": "conv_abc"
}
```

`userId` + `conversationId` together form the session key for `MemorySaver`. Same pair = same conversation thread, preserving full history.

Non-streaming response:

```json
{
  "id": "chatcmpl-xxxx",
  "object": "chat.completion",
  "created": 1234567890,
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "stop"
    }
  ]
}
```

Streaming response (`"stream": true`): returns SSE chunks in the same format as OpenAI streaming.

---

## Setup & Run

**1. Clone and create virtual environment**
```bash
git clone https://github.com/zchenpy/langGraphChatbot.git
cd langGraphChatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure environment**

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
LANGCHAIN_API_KEY=ls-...       # optional, for LangSmith tracing
```

**3. Choose your LLM backend**

In `llms.py`, set:
```python
DEFAULT_LLM_TYPE = "openai"    # use OpenAI gpt-4o-mini
# or
DEFAULT_LLM_TYPE = "ollama"    # use local Ollama model
```

For Ollama: install [Ollama](https://ollama.com), then run:
```bash
ollama pull deepseek-r1:14b
ollama serve
```

**4. Start the API server**
```bash
python3 main.py
# Server starts on http://0.0.0.0:8012
```

**5. Start the Web UI (optional)**
```bash
# in a separate terminal
python3 webUI.py
# UI available at http://127.0.0.1:7860
```

---

## Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI + uvicorn |
| Agent Framework | LangGraph (`StateGraph`, `MemorySaver`) |
| LLM (cloud) | OpenAI `gpt-4o-mini` via `langchain_openai` |
| LLM (local) | Ollama `deepseek-r1:14b` via `langchain_community` |
| Streaming | Server-Sent Events (`StreamingResponse`) |
| Observability | LangSmith (`LANGCHAIN_TRACING_V2`) |
| Web UI | Gradio |
| Config | `python-dotenv` |

---

## Design Decisions

**Why LangGraph over plain LangChain?**
LangGraph provides explicit state management and a compiled graph structure, making conversation flow easier to inspect, test, and extend. The `MemorySaver` checkpointer enables persistent session state without an external database.

**Why OpenAI-compatible schema?**
Adopting the `/v1/chat/completions` format means any client, tool, or frontend built against the OpenAI API works out of the box — no custom client code needed.

**Why externalized prompt templates?**
Keeping prompts outside application code separates concerns: engineers can iterate on prompts independently of the service logic, and templates can be version-controlled or swapped at deploy time.

1.  You need create a .env to set up your api keys like:
(openai api):
OPENAI_API_KEY=abcd 

(langsmith api)
LANGCHAIN_API_KEY=abcd

2. cd to 01_CHATBOT folder
   Create a virtual environment:
   python3 -m venv .venv

   Activate the virtual environment:
   source .venv/bin/activate

   Install the dependencies:
   pip install -r requirements.txt

3. To change the default model, modify the following line in llms.py:
   Set and switch LLM types
   Default to OpenAI (can be changed to "ollama")
   DEFAULT_LLM_TYPE = "openai"  # Change "openai" to "ollama" if needed
   
   Setup your local deepseek model: check reference video

4. Run the main:
   open terminal
   python3 main.py

5. Run the webUI:
   open another terminal
   python3 webUI.py
   http://127.0.0.1:7860/

Reference: https://github.com/NanGePlus/LangGraphChatBot

   
