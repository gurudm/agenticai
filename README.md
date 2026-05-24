# Banking Intelligence Agent API

A production-grade agentic RAG system for analyzing TD Bank Q1 2026 earnings transcripts.
Built with FastAPI, LangGraph, LangChain, and Anthropic Claude.

## Architecture

```
User Question
     │
     ▼
FastAPI Endpoint
     │
     ▼
LangGraph React Agent
     │
     ├── search_documents (RAG + ChromaDB)
     ├── calculate_ratio (financial calculations)
     └── identify_risks (risk analysis)
     │
     ▼
MemorySaver (per-session conversation memory)
     │
     ▼
Structured JSON Response
```

## Features
- Agentic Q&A over real earnings transcripts
- Semantic search with relevance threshold filtering
- Financial ratio calculation
- Risk factor identification
- Per-session conversation memory
- Tools used transparency in every response
- Health check endpoint

## Tech Stack
- Python 3.13
- FastAPI + Uvicorn
- LangGraph + LangChain
- Anthropic Claude Sonnet
- ChromaDB vector store
- HuggingFace embeddings
- PyPDF document loader

## Setup
```bash
pip install fastapi uvicorn langchain langchain-anthropic \
            langchain-chroma langchain-huggingface \
            langgraph pypdf chromadb duckduckgo-search

export CLAUDEKEY="your-api-key"
export RELEVANCE_THRESHOLD=1.8

uvicorn banking_agent_api:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Capabilities overview |
| GET | /health | System health check |
| POST | /ask | Ask the agent a question |
| DELETE | /session/{id} | Clear session |

## Example Response
```json
{
  "session_id": "analyst_1",
  "question": "What did Sona say about ROE targets?",
  "answer": "Based on page 10 of the transcript...",
  "tools_used": ["search_documents"]
}
```