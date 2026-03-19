from langchain_core.prompts import PromptTemplate


class PromptBuilder:
    """Builds the universal intent-based master prompt."""

    # 2. Define the universal schema-agnostic instruction set
    universal_decoupled_template = """Task: Generate a Cypher statement to query a medical knowledge graph.

    Instructions:
    1. INTENT-TO-SCHEMA MAPPING (CRITICAL): Read the user's question to identify the core concepts (e.g., medications, diseases, symptoms). Analyze the provided {schema} to find the exact Node Labels and Relationship Types that best represent those concepts. Do not assume labels; find the closest semantic match.

    2. UNIVERSAL SYNONYM EXPANSION: For ANY medical entity mentioned (drug, disease, pathogen), generate an array of synonyms, generic names, and brand names. Use the Cypher IN operator.
    - Example: WHERE toLower(n.name) IN ['diabetes', 'type 2 diabetes', 't2d']

    3. CATEGORICAL VS. FREE-TEXT SEARCH: Look at the properties available in the {schema}.
    - CATEGORICAL: If asking for a risk or severity level, use the IN operator with synonyms. Example: WHERE toLower(r.severity) IN ['major', 'severe', 'critical']
    - FREE-TEXT: If asking for a mechanism or symptom, use the CONTAINS operator. Example: WHERE toLower(r.mechanism) CONTAINS 'bleeding'

    4. RETURN EVERYTHING (CRITICAL): Return the target node AND the entire dictionary of relationship properties using the Cypher `properties()` function. Also, return the labels of the nodes so the frontend knows what type of entity it is.
    Example: RETURN labels(target_node) AS NodeType, target_node.name AS TargetName, properties(r) AS EdgeDetails

    5. DIRECTIONALITY MATTERS: 
    - Mutual relationships (e.g., drug interactions): Omit arrows (n1)-[r]-(n2)
    - Directed actions (e.g., treating a diagnosis): Omit arrows to ensure a match regardless of import direction, UNLESS the schema explicitly requires it.

    6. CASE INSENSITIVITY: ALWAYS use the toLower() function on string comparisons.

    Schema:
    {schema}

    Question: {question}
    Cypher Query:"""

    @staticmethod
    def get_prompt() -> PromptTemplate:
        return PromptTemplate(
            input_variables=["schema", "question"],
            template=PromptBuilder.universal_decoupled_template
        )
