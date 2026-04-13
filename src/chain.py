"""
Chain configuration for the Universal Medical Graph Explorer.
"""
from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain
from src.graph import get_graph
from src.prompt_builder import PromptBuilder


def build_qa_chain():
    """Initializes the LLMs and returns the Cypher Chain, Summary Chain, and Graph."""
    # 1. Setup the Graph and standard GPT-4o for complex Cypher generation
    graph = get_graph()
    cypher_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    cypher_prompt = PromptBuilder.get_prompt()

    cypher_chain = GraphCypherQAChain.from_llm(
        cypher_llm=cypher_llm,
        qa_llm=cypher_llm,
        graph=graph,
        verbose=True,  # Prints the dynamic Cypher to your terminal for debugging
        cypher_prompt=cypher_prompt,
        return_direct=True,  # Critical: Returns raw JSON instead of conversational summary
        validate_cypher=True,
        allow_dangerous_requests=True,  # Add this required security flag!
        top_k=100  # Override LangChain's default limit of 10
    )

    # 2. Setup the faster/cheaper GPT-4o-mini for the simple Text Summary
    summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    summary_prompt = PromptBuilder.get_summary_prompt()

    # Use LangChain Expression Language (LCEL) to pipe the prompt into the LLM
    summary_chain = summary_prompt | summary_llm

    # Return all three tools so the API can orchestrate them
    return cypher_chain, summary_chain, graph
