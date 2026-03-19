"""
FastAPI bridge connecting the frontend to the dynamic graph backend.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.chain import build_qa_chain

app = FastAPI(title="Universal Medical Graph API")

# Initialize chain and graph globally
chain, graph = build_qa_chain()


class QueryRequest(BaseModel):
    question: str


@app.post("/api/query")
async def execute_dynamic_query(request: QueryRequest):
    try:
        # Pull the absolute latest schema immediately before invoking the chain
        graph.refresh_schema()

        # Execute query; chain returns raw JSON because return_direct=True
        # The {schema} variable is injected automatically by LangChain
        response = chain.invoke({"query": request.question})
        raw_graph_data = response.get("result", [])

        # Failsafe: Query succeeded, but no matching paths exist in the graph
        if not raw_graph_data:
            return {
                "status": "empty",
                "message": "No matching data found in the current graph. Please try rephrasing or check clinical guidelines.",
                "data": []
            }

        # Success: Return the dynamic JSON array to React
        return {
            "status": "success",
            "message": "Graph data retrieved successfully.",
            "data": raw_graph_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error",
                    "message": "Failed to parse the medical query: " + str(e), "data": []}
        )
