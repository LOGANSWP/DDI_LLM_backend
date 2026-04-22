"""
FastAPI bridge connecting the frontend to the dynamic graph backend.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.chain import build_qa_chain
from src.graph import fetch_expanded_node, fetch_initial_graph

app = FastAPI(title="Drug to Drug Interaction LLM API")

# Using ["*"] allows requests from ANY origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize chain and graph globally
cypher_chain, summary_chain, graph = build_qa_chain()


class QueryRequest(BaseModel):
    question: str


class ExpandRequest(BaseModel):
    node_name: str


@app.post("/api/query")
async def execute_dynamic_query(request: QueryRequest):
    try:
        # Pull the absolute latest schema immediately before invoking the chain
        graph.refresh_schema()

        # PASS 1: Get raw JSON
        # Execute query; chain returns raw JSON because return_direct=True
        # The {schema} variable is injected automatically by LangChain
        response = cypher_chain.invoke({"query": request.question})
        raw_graph_data = response.get("result", [])

        # Failsafe: Query succeeded, but no matching paths exist in the graph
        if not raw_graph_data:
            return {
                "status": "empty",
                "message": "No matching data found.",
                "summary": "We could not find any records in the database matching your query.",
                "data": []
            }

        # PASS 2: Get text summary using summary_chain
        summary_response = summary_chain.invoke({
            "question": request.question,
            "data": str(raw_graph_data)
        })

        # Success: Return the dynamic JSON array to React
        return {
            "status": "success",
            "message": "Graph data retrieved successfully.",
            "summary": summary_response.content,
            "data": raw_graph_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to parse the medical query.",
                "summary": str(e),
                "data": []
            }
        )

# ==========================================
# Initial Graph Load (Top 10 Drugs)
# ==========================================


@app.get("/api/graph/init")
async def get_initial_graph():
    """Fetches the top 10 most connected drugs to act as root nodes."""
    try:
        # Calls the clean helper function from graph.py
        raw_graph_data = fetch_initial_graph(graph)
        return {
            "status": "success",
            "message": "Initial graph loaded with top 10 drug nodes and top 5 diagnosis nodes.",
            "data": raw_graph_data
        }
    except Exception as e:
        # Error Handling
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"Failed to load initial graph: {str(e)}",
                "data": []
            }
        )

# ==========================================
# Expand Node
# ==========================================


@app.post("/api/graph/expand")
async def expand_node(request: ExpandRequest):
    """Fetches all immediate neighbors (and their connecting edges) for a specific node."""
    try:
        # Calls the clean helper function from graph.py
        raw_graph_data = fetch_expanded_node(graph, request.node_name)

        return {
            "status": "success",
            "message": f"Expanded node: {request.node_name}",
            "data": raw_graph_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"Failed to expand node: {str(e)}",
                "data": []
            }
        )
