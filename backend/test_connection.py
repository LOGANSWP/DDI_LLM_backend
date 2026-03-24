# simple script to test neo4j connection without modifying data
from src.graph import get_graph

try:
    print("connecting to neo4j aura...")
    graph = get_graph()

    print("fetching schema...")
    schema = graph.schema

    print("\n connection successful!")
    print("\nschema preview:")
    print(schema[:500] + "..." if len(schema) > 500 else schema)

except Exception as e:
    print(f"\n connection failed: {e}")
