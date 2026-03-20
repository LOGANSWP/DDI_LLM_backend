import pytest
from langchain_core.prompts import PromptTemplate
from src.prompt_builder import PromptBuilder


def test_get_prompt_returns_correct_type():
    """Ensure the builder returns a valid LangChain PromptTemplate."""
    prompt = PromptBuilder.get_prompt()
    assert isinstance(prompt, PromptTemplate)


def test_prompt_input_variables():
    """Ensure the prompt expects exactly 'schema' and 'question'."""
    prompt = PromptBuilder.get_prompt()

    assert "schema" in prompt.input_variables
    assert "question" in prompt.input_variables
    assert len(prompt.input_variables) == 2


def test_prompt_contains_critical_instructions():
    """Check if the critical architectural rules are present in the template."""
    # We access the class attribute directly to verify the text
    template_text = PromptBuilder._build_template()

    # Verify the AI is being told to map intents
    assert "INTENT-TO-SCHEMA MAPPING" in template_text

    # Verify the AI is being told to generate synonyms
    assert "UNIVERSAL SYNONYM EXPANSION" in template_text

    # Verify the AI is instructed to return all edge properties dynamically
    assert "properties(r)" in template_text

    # Verify the safety rule for case insensitivity is present
    assert "toLower()" in template_text


def test_prompt_formatting_success():
    """Test that the prompt can successfully format given dummy inputs."""
    prompt = PromptBuilder.get_prompt()

    dummy_schema = "Nodes: [Patient, Drug] | Edges: [PRESCRIBED]"
    dummy_question = "What was the patient prescribed?"

    # Actually inject the variables just like LangChain will do in production
    formatted_prompt = prompt.format(
        schema=dummy_schema, question=dummy_question)

    # Ensure the inputs were successfully injected into the final string
    assert dummy_schema in formatted_prompt
    assert dummy_question in formatted_prompt
