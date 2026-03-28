import os
import sys
from unittest.mock import MagicMock, patch
import pytest

# 1. Inject dummy environment variables BEFORE any LangChain/Neo4j imports happen.
os.environ["OPENAI_API_KEY"] = "fake-api-key-for-testing"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "testpassword"

# 2. Create mock objects for the chain and the graph globally so tests can assert them
mock_chain = MagicMock()
mock_graph = MagicMock()

# 3. Use a pytest fixture to safely load the app and test client.


@pytest.fixture
def client():
    # Intercept the 'src.chain' import
    mock_src_chain = MagicMock()
    mock_src_chain.build_qa_chain.return_value = (mock_chain, mock_graph)

    # patch.dict safely applies the mock ONLY for the duration of the API tests
    with patch.dict('sys.modules', {'src.chain': mock_src_chain}):
        # Clear src.api from the cache if it exists so it re-imports using our fake chain
        sys.modules.pop('src.api', None)

        from src.api import app
        from fastapi.testclient import TestClient

        # 'yield' hands the client to the test.
        # When the test finishes, the 'with' block ends and sys.modules returns to normal!
        yield TestClient(app)


def test_query_endpoint_success(client):
    """Test scenario where the graph successfully finds matching data."""
    mock_chain.invoke.return_value = {"result": [
        {"d.name": "Metformin", "effect": "Hypoglycemia"}]}

    response = client.post(
        "/api/query", json={"question": "What interacts with Metformin?"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == "Graph data retrieved successfully."
    assert len(data["data"]) == 1

    mock_graph.refresh_schema.assert_called()
    mock_chain.invoke.assert_called_with(
        {"query": "What interacts with Metformin?"})


def test_query_endpoint_empty_results(client):
    """Test scenario where the query runs successfully but no nodes match the user's question."""
    mock_chain.invoke.return_value = {"result": []}

    response = client.post(
        "/api/query", json={"question": "What interacts with a fake drug?"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "empty"
    assert "No matching data found" in data["message"]
    assert data["data"] == []


def test_query_endpoint_error_handling(client):
    """Test scenario where LangChain or the database throws an error."""
    mock_chain.invoke.side_effect = Exception("OpenAI API timeout")

    response = client.post("/api/query", json={"question": "Test error?"})

    assert response.status_code == 500

    # FastAPI wraps HTTPException details inside a "detail" key
    data = response.json()["detail"]
    assert data["status"] == "error"
    assert "Failed to parse the medical query: OpenAI API timeout" in data["message"]


def test_get_initial_graph_success(client):
    """Test the initial load endpoint returns the top 10 root nodes correctly."""
    # Mock the pure Cypher query response from the graph driver
    mock_graph.query.return_value = [
        {
            "NodeType1": ["Drug"],
            "Target1": "Metformin",
            "NodeType2": None,  # Python's None becomes JSON's null
            "Target2": None,
            "EdgeDetails": None,
            "EdgeType": None
        }
    ]

    # Hit the new GET endpoint
    response = client.get("/api/graph/init")

    # Assert HTTP Status
    assert response.status_code == 200

    # Assert JSON payload matches our API docs
    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == "Initial graph loaded with 10 root nodes."
    assert len(data["data"]) == 1

    # Assert the clever "null" trick is working properly
    assert data["data"][0]["Target1"] == "Metformin"
    assert data["data"][0]["Target2"] is None

    # Verify the backend actually tried to run a Cypher query
    mock_graph.query.assert_called_once()


def test_get_initial_graph_error(client):
    """Test the initial load endpoint properly handles database errors."""
    # Force the graph driver to throw an exception
    mock_graph.query.side_effect = Exception("Database connection lost")

    response = client.get("/api/graph/init")

    # Assert it returns an HTTP 500 Failsafe
    assert response.status_code == 500

    # Assert it matches the Global Error Handling structure (inside "detail")
    data = response.json()["detail"]
    assert data["status"] == "error"
    assert "Failed to load initial graph: Database connection lost" in data["message"]
    assert data["data"] == []
