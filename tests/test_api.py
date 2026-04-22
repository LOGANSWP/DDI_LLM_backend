import os
import sys
from unittest.mock import MagicMock, patch, ANY
import pytest

# 1. Inject dummy environment variables BEFORE any LangChain/Neo4j imports happen.
os.environ["OPENAI_API_KEY"] = "fake-api-key-for-testing"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "testpassword"

# 2. Create mock objects for the chains and the graph globally so tests can assert them
mock_cypher_chain = MagicMock()
mock_summary_chain = MagicMock()
mock_graph = MagicMock()

# 3. Use an autouse fixture to safely wipe the mock states between EVERY test


@pytest.fixture(autouse=True)
def reset_mocks_between_tests():
    mock_cypher_chain.reset_mock()
    mock_summary_chain.reset_mock()
    mock_graph.reset_mock()

    mock_cypher_chain.invoke.side_effect = None
    mock_summary_chain.invoke.side_effect = None
    mock_graph.query.side_effect = None

    mock_cypher_chain.invoke.return_value = None
    mock_summary_chain.invoke.return_value = None
    mock_graph.query.return_value = None


@pytest.fixture
def client():
    # Intercept the 'src.chain' import
    mock_src_chain = MagicMock()

    # CRITICAL UPDATE: The refactored build_qa_chain now returns THREE items!
    mock_src_chain.build_qa_chain.return_value = (
        mock_cypher_chain, mock_summary_chain, mock_graph)

    # patch.dict safely applies the mock ONLY for the duration of the API tests
    with patch.dict('sys.modules', {'src.chain': mock_src_chain}):
        # Clear src.api from the cache if it exists so it re-imports using our fake chain
        sys.modules.pop('src.api', None)

        from src.api import app
        from fastapi.testclient import TestClient

        # 'yield' hands the client to the test.
        yield TestClient(app)


def test_query_endpoint_success(client):
    """Test scenario where the graph successfully finds matching data and generates a summary."""
    # Mock Pass 1: The Cypher Chain returns JSON
    mock_cypher_chain.invoke.return_value = {"result": [
        {"d.name": "Metformin", "effect": "Hypoglycemia"}]}

    # Mock Pass 2: The Summary Chain returns a generated paragraph (.content)
    mock_summary_chain.invoke.return_value = MagicMock(
        content="This is a mocked medical summary.")

    response = client.post(
        "/api/query", json={"question": "What interacts with Metformin?"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == "Graph data retrieved successfully."
    assert data["summary"] == "This is a mocked medical summary."
    assert len(data["data"]) == 1

    mock_graph.refresh_schema.assert_called()
    mock_cypher_chain.invoke.assert_called_with(
        {"query": "What interacts with Metformin?"})


def test_query_endpoint_empty_results(client):
    """Test scenario where the query runs successfully but no nodes match the user's question."""
    mock_cypher_chain.invoke.return_value = {"result": []}

    response = client.post(
        "/api/query", json={"question": "What interacts with a fake drug?"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "empty"
    assert "No matching data found" in data["message"]
    assert "We could not find any records" in data["summary"]
    assert data["data"] == []


def test_query_endpoint_error_handling(client):
    """Test scenario where LangChain or the database throws an error."""
    mock_cypher_chain.invoke.side_effect = Exception("OpenAI API timeout")

    response = client.post("/api/query", json={"question": "Test error?"})

    assert response.status_code == 500

    # FastAPI wraps HTTPException details inside a "detail" key
    data = response.json()["detail"]
    assert data["status"] == "error"
    assert "Failed to parse the medical query" in data["message"]
    # FIX: The actual error string was moved to the 'summary' field!
    assert "OpenAI API timeout" in data["summary"]


def test_get_initial_graph_success(client):
    """Test the initial load endpoint returns the top root nodes correctly."""
    # Mock the pure Cypher query returning both a Drug and a Diagnosis
    mock_graph.query.return_value = [
        {
            "NodeType1": ["Drug"],
            "Target1": "Metformin",
            "NodeType2": None,
            "Target2": None,
            "EdgeDetails": None,
            "EdgeType": None
        },
        {
            "NodeType1": ["Diagnosis"],
            "Target1": "Type 2 Diabetes",
            "NodeType2": None,
            "Target2": None,
            "EdgeDetails": None,
            "EdgeType": None
        }
    ]

    response = client.get("/api/graph/init")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    # Assert that it successfully processes our mixed list of 2 items
    assert len(data["data"]) == 2
    assert data["data"][0]["Target1"] == "Metformin"
    assert data["data"][0]["NodeType1"] == ["Drug"]
    assert data["data"][1]["Target1"] == "Type 2 Diabetes"
    assert data["data"][1]["NodeType1"] == ["Diagnosis"]

    mock_graph.query.assert_called_once()


def test_get_initial_graph_error(client):
    """Test the initial load endpoint properly handles database errors."""
    mock_graph.query.side_effect = Exception("Database connection lost")

    response = client.get("/api/graph/init")

    assert response.status_code == 500
    data = response.json()["detail"]
    assert data["status"] == "error"
    assert "Database connection lost" in data["message"]
    assert data["data"] == []


def test_expand_node_success(client):
    """Test the expand endpoint successfully returns neighbors for a clicked node."""
    mock_graph.query.return_value = [
        {
            "NodeType1": ["Drug"],
            "Target1": "Aspirin",
            "NodeType2": ["Drug"],
            "Target2": "Warfarin",
            "EdgeDetails": {"severity": "High"},
            "EdgeType": "INTERACTS_WITH"
        }
    ]

    response = client.post("/api/graph/expand", json={"node_name": "Aspirin"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == "Expanded node: Aspirin"
    assert len(data["data"]) == 1
    assert data["data"][0]["Target2"] == "Warfarin"

    # Verify the backend actually tried to run a Cypher query with parameters
    mock_graph.query.assert_called_with(
        ANY,  # Ignores checking the exact Cypher string from graph.py
        params={"node_name": "Aspirin"}
    )


def test_expand_node_error(client):
    """Test the expand endpoint properly handles database errors."""
    mock_graph.query.side_effect = Exception("Database timeout")

    response = client.post("/api/graph/expand", json={"node_name": "Aspirin"})

    assert response.status_code == 500
    data = response.json()["detail"]
    assert data["status"] == "error"
    assert "Database timeout" in data["message"]
    assert data["data"] == []
