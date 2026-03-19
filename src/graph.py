"""
Core graph connection module.
Requires zero hardcoded configuration, connecting directly to Neo4j to pull the live schema.
"""
import os
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv

load_dotenv()


def get_graph() -> Neo4jGraph:
    """Initialize the Neo4j graph connection using environment variables."""
    return Neo4jGraph(
        url=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        username=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "testpassword"),
    )
