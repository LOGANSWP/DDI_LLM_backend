import os
import pytest
from unittest.mock import patch, MagicMock, call

# 1. Inject dummy environment variables at the absolute top.
os.environ["OPENAI_API_KEY"] = "fake-api-key-for-testing"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "testpassword"


@pytest.fixture
def mock_chain_dependencies():
    # We patch the specific classes/functions imported inside src.chain
    with patch('src.chain.ChatOpenAI') as MockChat, \
            patch('src.chain.get_graph') as mock_get_graph, \
            patch('src.chain.PromptBuilder') as MockPromptBuilder, \
            patch('src.chain.GraphCypherQAChain') as MockQAChain:

        # Mock the main LLM and Graph
        fake_llm = MagicMock()
        MockChat.return_value = fake_llm

        fake_graph = MagicMock()
        mock_get_graph.return_value = fake_graph

        # Mock the Cypher Chain
        fake_cypher_chain = MagicMock()
        MockQAChain.from_llm.return_value = fake_cypher_chain

        # Mock the new Summary Prompt for the LangChain pipe (|)
        fake_summary_prompt = MagicMock()
        MockPromptBuilder.get_summary_prompt.return_value = fake_summary_prompt

        yield {
            "MockChat": MockChat,
            "mock_get_graph": mock_get_graph,
            "MockQAChain": MockQAChain,
            "fake_cypher_chain": fake_cypher_chain,
            "fake_graph": fake_graph
        }


def test_build_qa_chain_returns_tuple(mock_chain_dependencies):
    from src.chain import build_qa_chain

    # FIX: Unpack the THREE items that are now returned!
    cypher_chain, summary_chain, graph = build_qa_chain()

    # Verify it returns our fake chains and fake graph
    assert cypher_chain == mock_chain_dependencies["fake_cypher_chain"]
    assert graph == mock_chain_dependencies["fake_graph"]
    assert summary_chain is not None


def test_build_qa_chain_configuration(mock_chain_dependencies):
    from src.chain import build_qa_chain

    build_qa_chain()

    MockChat = mock_chain_dependencies["MockChat"]

    # FIX: Verify ChatOpenAI was called TWICE: Once for gpt-4o, once for gpt-4o-mini
    assert MockChat.call_count == 2
    MockChat.assert_has_calls([
        call(model="gpt-4o", temperature=0),
        call(model="gpt-4o-mini", temperature=0)
    ], any_order=True)

    # Verify the LangChain QA Chain was configured securely
    MockQAChain = mock_chain_dependencies["MockQAChain"]
    _, kwargs = MockQAChain.from_llm.call_args

    assert kwargs["return_direct"] is True
    assert kwargs["validate_cypher"] is True
    assert kwargs["top_k"] == 100  # Test the new limit
    assert kwargs["graph"] == mock_chain_dependencies["fake_graph"]
