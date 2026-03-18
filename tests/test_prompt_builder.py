"""
unit tests for prompt builder

tests prompt construction, schema injection, and few-shot example formatting.
"""

import pytest
from src.prompt_builder import PromptBuilder, PromptValidator, PromptComponents
from src.mock_schema import MOCK_SCHEMA


class TestPromptBuilder:
    """test basic prompt building functionality"""

    def setup_method(self):
        """create prompt builder for each test"""
        self.builder = PromptBuilder()

    def test_build_basic_prompt(self):
        """build prompt with user question"""
        question = "What drugs interact with Warfarin?"
        prompt = self.builder.build_prompt(question)

        assert len(prompt) > 0
        assert question in prompt
        assert "Cypher" in prompt or "cypher" in prompt

    def test_prompt_contains_schema(self):
        """prompt must include schema description"""
        question = "Find major interactions"
        prompt = self.builder.build_prompt(question)

        # check for node labels
        assert "Drug" in prompt
        assert "Diagnosis" in prompt

        # check for relationship types
        assert "INTERACTS_WITH" in prompt
        assert "TREATS" in prompt

    def test_prompt_contains_system_instructions(self):
        """prompt must contain system instructions"""
        question = "Test question"
        prompt = self.builder.build_prompt(question)

        # check for key instruction elements
        assert "read-only" in prompt.lower() or "match" in prompt.lower()
        assert "schema" in prompt.lower()

    def test_prompt_with_examples(self):
        """prompt with examples should include few-shot demonstrations"""
        question = "Show interactions"
        prompt = self.builder.build_prompt(question, include_examples=True)

        # should contain example questions and cypher
        assert "Question" in prompt or "question" in prompt
        examples = MOCK_SCHEMA.get_example_queries()

        # at least one example should be present
        assert any(ex['question'] in prompt for ex in examples)

    def test_prompt_without_examples(self):
        """prompt without examples should be shorter"""
        question = "Show interactions"
        prompt_with = self.builder.build_prompt(question, include_examples=True)
        prompt_without = self.builder.build_prompt(question, include_examples=False)

        assert len(prompt_without) < len(prompt_with)

        # example queries should not be in the no-examples version
        examples = MOCK_SCHEMA.get_example_queries()
        for example in examples:
            assert example['question'] not in prompt_without

    def test_prompt_contains_constraints(self):
        """prompt must include query constraints"""
        question = "Test"
        prompt = self.builder.build_prompt(question)

        # check for key constraints
        assert "toLower" in prompt or "case" in prompt.lower()
        assert "bidirectional" in prompt.lower() or "arrow" in prompt.lower()

    def test_prompt_structure_complete(self):
        """prompt should have all necessary sections"""
        question = "What interacts with Aspirin?"
        prompt = self.builder.build_prompt(question)

        # must mention required cypher clauses
        assert "MATCH" in prompt.upper()
        assert "RETURN" in prompt.upper()

        # must forbid write operations
        assert "CREATE" in prompt or "write" in prompt.lower()

    def test_different_questions_produce_different_prompts(self):
        """different questions should be reflected in prompts"""
        q1 = "Find interactions with Warfarin"
        q2 = "What treats diabetes?"

        prompt1 = self.builder.build_prompt(q1)
        prompt2 = self.builder.build_prompt(q2)

        assert q1 in prompt1
        assert q2 in prompt2
        assert q1 not in prompt2
        assert q2 not in prompt1

    def test_empty_question_handling(self):
        """should handle empty question gracefully"""
        prompt = self.builder.build_prompt("")
        # should still build a prompt, just with empty question
        assert "Schema" in prompt or "schema" in prompt

    def test_special_characters_in_question(self):
        """should handle special characters in questions"""
        question = "What's the interaction between Drug-A and Drug-B?"
        prompt = self.builder.build_prompt(question)

        assert question in prompt
        assert len(prompt) > 0


class TestPromptComponents:
    """test individual prompt components"""

    def setup_method(self):
        self.builder = PromptBuilder()

    def test_get_components(self):
        """get_components should return all parts"""
        question = "Test question"
        components = self.builder.get_components(question)

        assert isinstance(components, PromptComponents)
        assert components.user_question == question
        assert len(components.system_instructions) > 0
        assert len(components.schema_description) > 0
        assert len(components.examples) > 0

    def test_system_instructions_content(self):
        """system instructions should contain key directives"""
        components = self.builder.get_components("test")
        instructions = components.system_instructions.lower()

        # should mention read-only requirement
        assert "read-only" in instructions or "read only" in instructions

        # should forbid write operations
        assert "create" in instructions or "delete" in instructions

    def test_schema_description_from_mock(self):
        """schema description should come from mock schema"""
        components = self.builder.get_components("test")
        schema_desc = components.schema_description

        # should contain mock schema elements
        assert "Drug" in schema_desc
        assert "Diagnosis" in schema_desc
        assert "INTERACTS_WITH" in schema_desc

    def test_examples_from_mock_schema(self):
        """examples should come from mock schema"""
        components = self.builder.get_components("test")
        examples = components.examples

        mock_examples = MOCK_SCHEMA.get_example_queries()
        assert len(examples) == len(mock_examples)

        # check that examples have required structure
        for example in examples:
            assert "question" in example
            assert "cypher" in example
            assert len(example["question"]) > 0
            assert len(example["cypher"]) > 0

    def test_constraints_content(self):
        """constraints should specify rules"""
        components = self.builder.get_components("test")
        constraints = components.constraints.lower()

        # should mention key constraints
        assert "match" in constraints
        assert "return" in constraints
        assert "tolower" in constraints or "case" in constraints


class TestPromptValidator:
    """test prompt validation utilities"""

    def test_validate_complete_prompt(self):
        """complete prompt should pass validation"""
        builder = PromptBuilder()
        prompt = builder.build_prompt("What drugs interact?")

        is_valid = PromptValidator.validate_prompt(prompt)
        assert is_valid is True

    def test_validate_incomplete_prompt(self):
        """incomplete prompt should fail validation"""
        incomplete_prompt = "Just a random string"
        is_valid = PromptValidator.validate_prompt(incomplete_prompt)
        assert is_valid is False

    def test_validate_components_complete(self):
        """complete components should have no errors"""
        components = PromptComponents(
            system_instructions="You are a cypher expert",
            schema_description="Schema has Drug nodes and relationships",
            examples=[{"question": "test", "cypher": "MATCH (d) RETURN d"}],
            user_question="What interacts?",
            constraints="Use MATCH and RETURN only"
        )

        errors = PromptValidator.validate_components(components)
        assert len(errors) == 0

    def test_validate_components_missing_instructions(self):
        """missing system instructions should error"""
        components = PromptComponents(
            system_instructions="",
            schema_description="Schema info",
            examples=[],
            user_question="test",
            constraints="constraints"
        )

        errors = PromptValidator.validate_components(components)
        assert len(errors) > 0
        assert any("instruction" in err.lower() for err in errors)

    def test_validate_components_missing_schema(self):
        """missing schema should error"""
        components = PromptComponents(
            system_instructions="instructions",
            schema_description="",
            examples=[],
            user_question="test",
            constraints="constraints"
        )

        errors = PromptValidator.validate_components(components)
        assert len(errors) > 0
        assert any("schema" in err.lower() for err in errors)

    def test_validate_components_empty_question(self):
        """empty question should error"""
        components = PromptComponents(
            system_instructions="instructions",
            schema_description="Schema has nodes and relationships",
            examples=[],
            user_question="   ",
            constraints="constraints"
        )

        errors = PromptValidator.validate_components(components)
        assert len(errors) > 0
        assert any("question" in err.lower() for err in errors)

    def test_validate_components_missing_question(self):
        """missing question should error"""
        components = PromptComponents(
            system_instructions="instructions",
            schema_description="Schema has nodes and relationships",
            examples=[],
            user_question="",
            constraints="constraints"
        )

        errors = PromptValidator.validate_components(components)
        assert len(errors) > 0

    def test_validate_schema_has_nodes(self):
        """schema description should mention nodes or labels"""
        components = PromptComponents(
            system_instructions="instructions",
            schema_description="Just some random text",
            examples=[],
            user_question="test",
            constraints="constraints"
        )

        errors = PromptValidator.validate_components(components)
        assert any("node" in err.lower() or "label" in err.lower() for err in errors)

    def test_validate_schema_has_relationships(self):
        """schema description should mention relationships"""
        components = PromptComponents(
            system_instructions="instructions",
            schema_description="Has some labels",
            examples=[],
            user_question="test",
            constraints="constraints"
        )

        errors = PromptValidator.validate_components(components)
        assert any("relationship" in err.lower() for err in errors)


class TestPromptBuilderEdgeCases:
    """test edge cases and error handling"""

    def setup_method(self):
        self.builder = PromptBuilder()

    def test_very_long_question(self):
        """should handle very long questions"""
        long_question = "What are the " + "very " * 100 + "severe interactions?"
        prompt = self.builder.build_prompt(long_question)

        assert long_question in prompt
        assert len(prompt) > 0

    def test_question_with_newlines(self):
        """should handle questions with newlines"""
        question = "What drugs\ninteract with\nWarfarin?"
        prompt = self.builder.build_prompt(question)

        assert question in prompt

    def test_question_with_cypher_keywords(self):
        """should handle questions containing cypher keywords"""
        question = "MATCH drugs that CREATE interactions"
        prompt = self.builder.build_prompt(question)

        assert question in prompt
        # should still forbid actual CREATE operations
        assert "CREATE" in prompt  # in the forbidden list

    def test_unicode_characters(self):
        """should handle unicode in questions"""
        question = "What drugs interact with Café®?"
        prompt = self.builder.build_prompt(question)

        assert question in prompt

    def test_prompt_builder_with_custom_schema(self):
        """should work with custom schema object"""
        custom_builder = PromptBuilder(schema=MOCK_SCHEMA)
        prompt = custom_builder.build_prompt("test")

        assert len(prompt) > 0
        assert "Drug" in prompt
