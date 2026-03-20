"""
Chain configuration for the Universal Medical Graph Explorer.
"""
from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain
from src.graph import get_graph
from src.prompt_builder import PromptBuilder


def build_qa_chain():
    """Initializes the LLM to execute queries directly and return raw JSON data."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    graph = get_graph()
    prompt = PromptBuilder.get_prompt()

    chain = GraphCypherQAChain.from_llm(
        cypher_llm=llm,
        qa_llm=llm,
        graph=graph,
        verbose=True,  # Prints the dynamic Cypher to your terminal for debugging
        cypher_prompt=prompt,
        return_direct=True,  # Critical: Returns raw JSON instead of conversational summary
        validate_cypher=True,
        allow_dangerous_requests=True  # Add this required security flag!
    )
    return chain, graph
