"""
prompt builder for text-to-cypher generation

constructs prompts with schema injection and few-shot examples.
designed to minimize llm hallucination by providing explicit constraints.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass

from src.mock_schema import MOCK_SCHEMA


@dataclass
class PromptComponents:
    """structured components of a prompt"""
    system_instructions: str
    schema_description: str
    examples: List[Dict[str, str]]
    user_question: str
    constraints: str


class PromptBuilder:
    """
    builds prompts for cypher generation with schema injection

    uses few-shot prompting to teach the llm:
    - valid node labels and relationships
    - proper cypher syntax for this specific schema
    - case sensitivity rules
    - bidirectional relationship handling
    """

    def __init__(self, schema=MOCK_SCHEMA):
        self.schema = schema

    def build_prompt(self, user_question: str, include_examples: bool = True) -> str:
        """
        build complete prompt for cypher generation

        args:
            user_question: natural language question from user
            include_examples: whether to include few-shot examples

        returns:
            formatted prompt string ready for llm
        """
        components = self._get_prompt_components(user_question, include_examples)
        return self._format_prompt(components)

    def _get_prompt_components(
        self, user_question: str, include_examples: bool
    ) -> PromptComponents:
        """gather all components needed for the prompt"""

        system_instructions = self._get_system_instructions()
        schema_description = self.schema.get_schema_description()
        examples = self.schema.get_example_queries() if include_examples else []
        constraints = self._get_constraints()

        return PromptComponents(
            system_instructions=system_instructions,
            schema_description=schema_description,
            examples=examples,
            user_question=user_question,
            constraints=constraints
        )

    def _get_system_instructions(self) -> str:
        """get base system instructions for the llm"""
        return """You are a Cypher query generation expert for a drug interaction knowledge graph.

Your task is to convert natural language questions into valid, safe Cypher queries.

Critical requirements:
- Generate ONLY valid Cypher queries based on the provided schema
- Use ONLY the node labels, relationship types, and properties defined in the schema
- Generate read-only queries (MATCH + RETURN only)
- Never use CREATE, DELETE, MERGE, SET, DROP, or other write operations
- Always use toLower() for case-insensitive string matching
- For drug interactions, use bidirectional relationships without arrows: (d1)-[i:INTERACTS_WITH]-(d2)

Output format:
Return ONLY the Cypher query without any explanation, comments, or markdown formatting.
Do not include ```cypher``` tags or any other wrapper text.
"""

    def _get_constraints(self) -> str:
        """get additional constraints and rules"""
        return """
Query constraints:
1. Use only MATCH and RETURN clauses
2. All string comparisons must use toLower() for case insensitivity
3. Drug interaction relationships are bidirectional - omit directional arrows
4. Only query properties that exist in the schema
5. If the question cannot be answered with the available schema, return: "SCHEMA_INSUFFICIENT"
6. Keep queries simple and focused on the user's specific question
"""

    def _format_prompt(self, components: PromptComponents) -> str:
        """format all components into final prompt"""

        parts = [
            components.system_instructions,
            "",
            components.schema_description,
            "",
            components.constraints,
        ]

        # add few-shot examples if provided
        if components.examples:
            parts.append("")
            parts.append("Example queries:")
            parts.append("")
            for i, example in enumerate(components.examples, 1):
                parts.append(f"Question {i}: {example['question']}")
                parts.append(f"Cypher {i}:")
                parts.append(example['cypher'])
                parts.append("")

        # add user question
        parts.append("Now generate a Cypher query for this question:")
        parts.append(f"Question: {components.user_question}")
        parts.append("")
        parts.append("Cypher query:")

        return "\n".join(parts)

    def get_components(self, user_question: str) -> PromptComponents:
        """
        get prompt components separately for inspection/testing

        useful for debugging and validating prompt structure
        """
        return self._get_prompt_components(user_question, include_examples=True)


class PromptValidator:
    """validates that prompts contain necessary components"""

    @staticmethod
    def validate_prompt(prompt: str) -> bool:
        """
        check that prompt contains key required elements

        returns:
            true if prompt appears complete
        """
        required_elements = [
            "schema",  # must describe schema
            "cypher",  # must mention cypher
            "match",   # must explain MATCH clause requirement
            "return",  # must explain RETURN requirement
        ]

        prompt_lower = prompt.lower()
        return all(element in prompt_lower for element in required_elements)

    @staticmethod
    def validate_components(components: PromptComponents) -> List[str]:
        """
        validate prompt components

        returns:
            list of validation errors (empty if valid)
        """
        errors = []

        if not components.system_instructions:
            errors.append("missing system instructions")

        if not components.schema_description:
            errors.append("missing schema description")

        if not components.user_question:
            errors.append("missing user question")

        if not components.user_question.strip():
            errors.append("user question is empty or whitespace")

        # check that schema description contains key elements
        schema_lower = components.schema_description.lower()
        if "node" not in schema_lower and "label" not in schema_lower:
            errors.append("schema description missing node/label information")

        if "relationship" not in schema_lower:
            errors.append("schema description missing relationship information")

        return errors
