"""
FastAPI bridge connecting the frontend to the dynamic graph backend.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.chain import build_qa_chain

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
chain, graph = build_qa_chain()


class QueryRequest(BaseModel):
    question: str


class ExpandRequest(BaseModel):
    node_name: str


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

# ==========================================
# Initial Graph Load (Top 10 Drugs)
# ==========================================


@app.get("/api/graph/init")
async def get_initial_graph():
    """Fetches the top 10 most connected drugs to act as root nodes."""
    try:
        # Pure Cypher: Find top 10 drugs by degree, return ONLY the nodes.
        # We use 'null' for the edge/target2 columns to keep the JSON structure identical.
        cypher_query = """
        MATCH (d:Drug)
        WITH d ORDER BY COUNT { (d)--() } DESC LIMIT 10
        RETURN labels(d) AS NodeType1, d.name AS Target1, 
               null AS NodeType2, null AS Target2, 
               null AS EdgeDetails, null AS EdgeType
        """
        # Execute the pure cypher query directly against the database
        raw_graph_data = graph.query(cypher_query)

        return {
            "status": "success",
            "message": "Initial graph loaded with 10 root nodes.",
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
        # Only respond drugs interaction (case-insensitive)
        cypher_query = """
        MATCH (n)-[r]-(m:Drug)
        WHERE toLower(n.name) = toLower($node_name)
        RETURN labels(n) AS NodeType1, n.name AS Target1, 
               labels(m) AS NodeType2, m.name AS Target2, 
               properties(r) AS EdgeDetails, type(r) AS EdgeType
        LIMIT 50
        """
        # Pass the parameters dictionary to safely inject the user's clicked node
        raw_graph_data = graph.query(
            cypher_query, params={"node_name": request.node_name})

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
