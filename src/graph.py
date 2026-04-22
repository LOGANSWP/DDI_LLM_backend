"""
Core graph connection module.
Requires zero hardcoded configuration, connecting directly to Neo4j to pull the live schema.
"""
import os
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv

# Force reload the .env file
load_dotenv(override=True)


def get_graph() -> Neo4jGraph:
    """Initialize the Neo4j graph connection using environment variables."""
    return Neo4jGraph(
        url=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        username=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "testpassword"),
        database=os.environ.get("NEO4J_DATABASE", "neo4j")
    )


def fetch_initial_graph(graph: Neo4jGraph) -> list:
    """Executes pure Cypher to get the top 10 root drugs."""
    query = """
    MATCH (d:Drug)
    WITH d ORDER BY COUNT { (d)--(:Drug) } DESC LIMIT 10
    RETURN labels(d) AS NodeType1, d.name AS Target1, 
           null AS NodeType2, null AS Target2, 
           null AS EdgeDetails, null AS EdgeType
    """
    return graph.query(query)


def fetch_expanded_node(graph: Neo4jGraph, node_name: str) -> list:
    """Executes pure Cypher to get all valid neighbors (Drugs or Diagnoses) for a specific node."""
    query = """
    MATCH (n)-[r]-(m)
    WHERE toLower(COALESCE(n.name, n.long_title, "")) = toLower($node_name)
      AND COALESCE(m.name, m.long_title) IS NOT NULL
    RETURN labels(n) AS NodeType1, COALESCE(n.name, n.long_title) AS Target1, 
           labels(m) AS NodeType2, COALESCE(m.name, m.long_title) AS Target2, 
           properties(r) AS EdgeDetails, type(r) AS EdgeType
    """
    return graph.query(query, params={"node_name": node_name})
