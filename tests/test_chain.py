import os
import pytest
from unittest.mock import patch, MagicMock

# 1. Inject dummy environment variables at the absolute top.
# This prevents LangChain from crashing if it checks for keys during test collection.
os.environ["OPENAI_API_KEY"] = "fake-api-key-for-testing"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "testpassword"

# 2. We use a fixture to mock all the heavy external dependencies.


@pytest.fixture
def mock_chain_dependencies():
    # We patch the specific classes/functions imported inside src.chain
    with patch('src.chain.ChatOpenAI') as MockChat, \
            patch('src.chain.get_graph') as mock_get_graph, \
            patch('src.chain.PromptBuilder') as MockPromptBuilder, \
            patch('src.chain.GraphCypherQAChain') as MockQAChain:

        # Create fake instances that these mocks will return
        fake_llm = MagicMock()
        MockChat.return_value = fake_llm

        fake_graph = MagicMock()
        mock_get_graph.return_value = fake_graph

        fake_chain = MagicMock()
        MockQAChain.from_llm.return_value = fake_chain

        # Yield all the mocks so our tests can assert against them
        yield {
            "MockChat": MockChat,
            "mock_get_graph": mock_get_graph,
            "MockQAChain": MockQAChain,
            "fake_chain": fake_chain,
            "fake_graph": fake_graph
        }

# 3. Test that the function returns the correct objects


def test_build_qa_chain_returns_tuple(mock_chain_dependencies):
    # Local import to defeat the auto-formatter!
    from src.chain import build_qa_chain

    chain, graph = build_qa_chain()

    # Verify it returns our fake chain and fake graph
    assert chain == mock_chain_dependencies["fake_chain"]
    assert graph == mock_chain_dependencies["fake_graph"]

# 4. Test the CRITICAL configuration parameters


def test_build_qa_chain_configuration(mock_chain_dependencies):
    from src.chain import build_qa_chain

    build_qa_chain()

    # Verify OpenAI was initialized with GPT-4o and 0 temperature (for deterministic logic)
    MockChat = mock_chain_dependencies["MockChat"]
    MockChat.assert_called_once_with(model="gpt-4o", temperature=0)

    # Verify the LangChain QA Chain was configured securely and correctly
    MockQAChain = mock_chain_dependencies["MockQAChain"]

    # Extract the keyword arguments passed to from_llm
    _, kwargs = MockQAChain.from_llm.call_args

    # CRITICAL: If return_direct is False, the AI will try to chat instead of returning JSON data!
    assert kwargs["return_direct"] is True

    # CRITICAL: validate_cypher ensures the query doesn't crash the DB
    assert kwargs["validate_cypher"] is True

    # Ensure it's using the graph we provided
    assert kwargs["graph"] == mock_chain_dependencies["fake_graph"]
