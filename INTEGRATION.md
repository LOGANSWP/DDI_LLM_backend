# Integration Guide

This LLM backend service is designed to work as a standalone microservice.

## How to Use with DDI_Application

The DDI_Application repository contains the frontend and a minimal backend proxy that calls this service.

### Architecture

```
DDI_Application (Frontend + Proxy) → DDI_LLM_backend (This Service) → Neo4j
```

### Running Both Services

**Terminal 1 - Start this LLM Backend:**
```bash
cd DDI_LLM_backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn src.api:app --reload --port 8000
```

**Terminal 2 - Start DDI_Application:**
```bash
cd DDI_Application/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8001
```

**Terminal 3 - Start Frontend:**
```bash
cd DDI_Application/frontend
npm install
npm run dev
```

### Configuration

Both services need access to Neo4j credentials. You can:

**Option 1: Shared .env file**
Place `.env` in a parent directory accessible to both

**Option 2: Separate .env files**
Each repo has its own `.env` with Neo4j credentials

### API Endpoints

This LLM backend exposes:
- `POST /api/query` - Accepts natural language, returns graph data

The DDI_Application backend proxies this endpoint at the same path.

### Environment Variables

Required in this service:
```env
OPENAI_API_KEY=your_key
NEO4J_URI=neo4j+s://your_instance.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

### Development Workflow

1. Make changes to LLM logic, prompts, or schema handling in this repo
2. Test by calling this service directly on port 8000
3. Once working, DDI_Application will automatically use the changes
4. No changes needed in DDI_Application unless API contract changes
