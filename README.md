# Agentic AI Banking System

A production-grade agentic AI system for banking document analysis.
Built over 3 weeks as part of an intensive AI engineering learning program.

## What This System Does

Analyzes real bank earnings transcripts using:
- Semantic search over PDF documents
- Multi-tool AI agents that reason and act
- Conditional workflows that adapt based on findings
- Human-in-the-loop approval for high risk findings
- Per-session conversation memory

## Architecture Evolution

### Week 1 - Foundation
Claude API → Prompt Engineering → LangChain → Memory → FastAPI

### Week 2 - Intelligence  
ChromaDB RAG → PDF Loading → LangGraph Agent → Multi-tool Reasoning

### Week 3 - Workflows
LangGraph StateGraph → Conditional Routing → Human Approval → Workflow API

## Tech Stack
- Python 3.13
- FastAPI + Uvicorn
- LangChain + LangGraph
- Anthropic Claude Sonnet
- ChromaDB vector store
- HuggingFace embeddings
- PyPDF document loader
- MemorySaver checkpointing

## Key Features

### RAG System
Semantic search over TD Bank Q1 2026 earnings transcript with
relevance threshold filtering. Chunks at 1500 chars with 200 overlap.

### Agentic Tools
- search_documents - semantic search over earnings transcript
- calculate_ratio - financial ratio computation
- identify_risks - risk factor extraction

### Conditional Workflow
```
Question → Retrieve → Extract Metrics
                           │
              ┌────────────┼────────────┐
           High Risk    Medium/Low   No Data
              │            │            │
         Deep Risk    Standard      Insufficient
         Analysis     Analysis      Handler
              │            │
         Human Review  Generate Report
              │
         Approve/Reject → Final Report
```

### Human-in-the-Loop
High risk findings pause the workflow and require explicit
human approval via API before generating the final report.

## Setup
```bash
pip install fastapi uvicorn langchain langchain-anthropic \
            langchain-chroma langchain-huggingface \
            langgraph pypdf chromadb

export CLAUDEKEY="your-anthropic-api-key"
```

## Run

### Week 2 - Agent API
```bash
cd week2
uvicorn banking_agent_api:app --reload
```

### Week 3 - Workflow API
```bash
cd week3
uvicorn workflow_api:app --reload
```

## API Reference

### Workflow API (Week 3)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Health check |
| POST | /analyze | Start analysis workflow |
| POST | /review/{thread_id} | Submit human decision |
| GET | /status/{thread_id} | Check workflow status |

### Example Flow
```bash
# Start analysis
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the credit risks?", "thread_id": "session_1"}'

# If awaiting_review, approve it
curl -X POST http://localhost:8000/review/session_1 \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "session_1", "decision": "approve", "comments": "Risk acceptable"}'
```

## Author
Gurudarshan - Senior Software Engineer
13 years experience | Investment Banking Domain | Morgan Stanley
Toronto, Canada