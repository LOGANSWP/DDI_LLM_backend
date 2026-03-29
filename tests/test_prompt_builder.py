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
    template_text = PromptBuilder._build_template()

    assert "INTENT-TO-SCHEMA MAPPING" in template_text
    assert "UNIVERSAL SYNONYM EXPANSION" in template_text
    assert "properties(r)" in template_text
    assert "toLower()" in template_text


def test_prompt_formatting_success():
    """Test that the prompt can successfully format given dummy inputs."""
    prompt = PromptBuilder.get_prompt()

    dummy_schema = "Nodes: [Patient, Drug] | Edges: [PRESCRIBED]"
    dummy_question = "What was the patient prescribed?"

    formatted_prompt = prompt.format(
        schema=dummy_schema, question=dummy_question)

    assert dummy_schema in formatted_prompt
    assert dummy_question in formatted_prompt

# ==========================================
# NEW TESTS: The Text Summary Prompt
# ==========================================


def test_get_summary_prompt_returns_correct_type():
    """Ensure the summary prompt builder returns a valid LangChain PromptTemplate."""
    prompt = PromptBuilder.get_summary_prompt()
    assert isinstance(prompt, PromptTemplate)


def test_summary_prompt_input_variables():
    """Ensure the summary prompt expects exactly 'question' and 'data'."""
    prompt = PromptBuilder.get_summary_prompt()

    assert "question" in prompt.input_variables
    assert "data" in prompt.input_variables
    assert len(prompt.input_variables) == 2
