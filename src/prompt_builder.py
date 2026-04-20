from langchain_core.prompts import PromptTemplate


class PromptBuilder:
    """Builds the universal intent-based master prompt dynamically."""

    # The Core Task
    TASK_DESCRIPTION = "Task: Generate a Cypher statement to query a medical knowledge graph."

    # Individual Instructions (Notice: No hardcoded numbers)
    INST_MAPPING = (
        "INTENT-TO-SCHEMA MAPPING (CRITICAL): Read the user's question to identify the core concepts "
        "(e.g., medications, diseases, symptoms). Analyze the provided {schema} to find the exact Node "
        "Labels and Relationship Types that best represent those concepts. Do not assume labels; "
        "find the closest semantic match."
    )

    INST_SYNONYM = (
        "SYNONYMS & PARTIAL MATCHING: \n"
        "   - SPECIFIC ENTITIES: For specific drugs or exact diseases, generate synonyms and use the IN operator. "
        "Example: WHERE toLower(n.name) IN ['advil', 'ibuprofen']\n"
        "   - BROAD CATEGORIES: If the user asks for a general condition (e.g., 'kidney failure', 'cancer'), use the CONTAINS operator with key terms to catch all variations. "
        "Example: WHERE toLower(n.name) CONTAINS 'kidney' OR toLower(n.name) CONTAINS 'renal'"
    )

    INST_SEARCH_TYPE = (
        "CATEGORICAL, FREE-TEXT, & NUMERICAL SEARCH: Look at the properties available in the {schema}.\n"
        "   - CATEGORICAL: For strings with few options, use IN. Example: WHERE toLower(r.severity) IN ['major', 'severe']\n"
        "   - FREE-TEXT: For descriptions, use CONTAINS. Example: WHERE toLower(r.mechanism) CONTAINS 'bleeding'\n"
        "   - NUMERICAL/SORTING: If the user asks for 'most common', 'highest', or 'top', find the relevant numerical property in the schema (e.g., admission_count or avg_severity) and use ORDER BY and LIMIT. "
        "Example: ORDER BY r.admission_count DESC LIMIT 10"
    )

    INST_RETURN = (
        "STRICT RETURN FORMAT (CRITICAL): You MUST return exactly these six columns using these exact aliases, no matter what your MATCH clause looks like. Do not change these aliases:\n"
        "   1. labels(n) AS NodeType1  (Replace 'n' with your first node variable)\n"
        "   2. n.name AS Target1\n"
        "   3. labels(m) AS NodeType2  (Replace 'm' with your second node variable)\n"
        "   4. m.name AS Target2\n"
        "   5. properties(r) AS EdgeDetails (Replace 'r' with your relationship variable)\n"
        "   6. type(r) AS EdgeType\n"
        "   Example: RETURN labels(d1) AS NodeType1, d1.name AS Target1, labels(d2) AS NodeType2, d2.name AS Target2, properties(r) AS EdgeDetails, type(r) AS EdgeType"
    )

    INST_DIRECTION = (
        "DIRECTIONALITY MATTERS: \n"
        "   - Mutual relationships (e.g., drug interactions): Omit arrows (n1)-[r]-(n2)\n"
        "   - Directed actions (e.g., treating a diagnosis): Omit arrows to ensure a match regardless "
        "of import direction, UNLESS the schema explicitly requires it."
    )

    INST_CASE = "CASE INSENSITIVITY: ALWAYS use the toLower() function on string comparisons."

    # Footer
    FOOTER = "Schema:\n{schema}\n\nQuestion: {question}\nCypher Query:"

    # --- NEW: Summary Prompt ---
    SUMMARY_TEMPLATE = (
        "The user asked: '{question}'.\n"
        "The database returned this raw data: {data}\n"
        "Write a brief, professional, 1-to-2 paragraph medical summary explaining this data to the user. "
        "Do not use markdown lists. If the data is empty, state that no known interactions were found."
    )

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

    @classmethod
    def get_summary_prompt(cls) -> PromptTemplate:
        """Returns the prompt template for the Phase 2 text summary."""
        return PromptTemplate.from_template(cls.SUMMARY_TEMPLATE)
