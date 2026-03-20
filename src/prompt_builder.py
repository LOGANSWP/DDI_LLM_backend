from langchain_core.prompts import PromptTemplate


class PromptBuilder:
    """Builds the universal intent-based master prompt dynamically."""

    # 1. The Core Task
    TASK_DESCRIPTION = "Task: Generate a Cypher statement to query a medical knowledge graph."

    # 2. Individual Instructions (Notice: No hardcoded numbers)
    INST_MAPPING = (
        "INTENT-TO-SCHEMA MAPPING (CRITICAL): Read the user's question to identify the core concepts "
        "(e.g., medications, diseases, symptoms). Analyze the provided {schema} to find the exact Node "
        "Labels and Relationship Types that best represent those concepts. Do not assume labels; "
        "find the closest semantic match."
    )

    INST_SYNONYM = (
        "UNIVERSAL SYNONYM EXPANSION: For ANY medical entity mentioned (drug, disease, pathogen), "
        "generate an array of synonyms, generic names, and brand names. Use the Cypher IN operator.\n"
        "   - Example: WHERE toLower(n.name) IN ['diabetes', 'type 2 diabetes', 't2d']"
    )

    INST_SEARCH_TYPE = (
        "CATEGORICAL VS. FREE-TEXT SEARCH: Look at the properties available in the {schema}.\n"
        "   - CATEGORICAL: If asking for a risk or severity level, use the IN operator with synonyms. "
        "Example: WHERE toLower(r.severity) IN ['major', 'severe', 'critical']\n"
        "   - FREE-TEXT: If asking for a mechanism or symptom, use the CONTAINS operator. "
        "Example: WHERE toLower(r.mechanism) CONTAINS 'bleeding'"
    )

    INST_RETURN = (
        "RETURN EVERYTHING (CRITICAL): Return the target node AND the entire dictionary of relationship "
        "properties using the Cypher `properties()` function. Also, return the labels of the nodes so "
        "the frontend knows what type of entity it is.\n"
        "   Example: RETURN labels(target_node) AS NodeType, target_node.name AS TargetName, properties(r) AS EdgeDetails"
    )

    INST_DIRECTION = (
        "DIRECTIONALITY MATTERS: \n"
        "   - Mutual relationships (e.g., drug interactions): Omit arrows (n1)-[r]-(n2)\n"
        "   - Directed actions (e.g., treating a diagnosis): Omit arrows to ensure a match regardless "
        "of import direction, UNLESS the schema explicitly requires it."
    )

    INST_CASE = "CASE INSENSITIVITY: ALWAYS use the toLower() function on string comparisons."

    # 3. Footer
    FOOTER = "Schema:\n{schema}\n\nQuestion: {question}\nCypher Query:"

    @classmethod
    def _build_template(cls) -> str:
        """Assembles the modular parts and dynamically numbers the instructions."""

        # Group the active instructions in the exact order you want them to appear
        active_instructions = [
            cls.INST_MAPPING,
            cls.INST_SYNONYM,
            cls.INST_SEARCH_TYPE,
            cls.INST_RETURN,
            cls.INST_DIRECTION,
            cls.INST_CASE
        ]

        # Use enumerate to automatically add "1. ", "2. " based on the array index
        numbered_instructions = [
            f"{index}. {instruction}"
            for index, instruction in enumerate(active_instructions, start=1)
        ]

        # Join them together with double newlines for readability
        formatted_instructions = "\n\n".join(numbered_instructions)

        return f"{cls.TASK_DESCRIPTION}\n\nInstructions:\n{formatted_instructions}\n\n{cls.FOOTER}"

    @classmethod
    def get_prompt(cls) -> PromptTemplate:
        """Returns the assembled LangChain PromptTemplate."""
        return PromptTemplate(
            input_variables=["schema", "question"],
            template=cls._build_template()
        )
