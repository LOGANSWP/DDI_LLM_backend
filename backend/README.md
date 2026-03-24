# Backend - Drug Interaction LLM API

FastAPI backend with LangChain orchestration for text-to-Cypher conversion.

## Quick Start

### Prerequisites
- Python 3.13+
- Neo4j database (local or Aura cloud)
- OpenAI API key

### Setup

1. **Create virtual environment** (from backend directory):
```bash
cd backend
python -m venv venv
source venv/bin/activate  # mac/linux
# or
venv\Scripts\activate  # windows
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment** (in project root):
The `.env` file is in the parent directory and shared with frontend.

4. **Run tests**:
```bash
python -m pytest tests/ -v
```

5. **Start the server**:
```bash
uvicorn src.api:app --reload --port 8000
```

6. **Access API docs**:
Open `http://localhost:8000/docs`

## API Endpoints

### POST /api/query
Query the drug interaction graph using natural language.

**Request:**
```json
{
  "question": "What are severe drug interactions?"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Graph data retrieved successfully.",
  "data": [...]
}
```

## Architecture

- **src/api.py** - FastAPI endpoints with CORS
- **src/chain.py** - LangChain GraphCypherQAChain configuration
- **src/graph.py** - Neo4j connection
- **src/prompt_builder.py** - Intent-based prompt template

## Testing

Run all tests:
```bash
python -m pytest tests/ -v --cov
```

Run specific test file:
```bash
python -m pytest tests/test_api.py -v
```
