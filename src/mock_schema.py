"""
mock schema for drug interaction graph database

this file defines the temporary schema used during development.
when the real schema is ready, replace this entire file with the production schema.

design note: keep all schema definitions in one place for easy replacement.
"""

from typing import Dict, List, Set


class MockGraphSchema:
    """
    mock neo4j graph schema for drug-drug interactions

    represents a simplified version of the production schema.
    includes node labels, relationship types, properties, and validation rules.
    """

    # node labels allowed in the graph
    NODE_LABELS: Set[str] = {
        "Drug",
        "Diagnosis"
    }

    # relationship types allowed in the graph
    RELATIONSHIP_TYPES: Set[str] = {
        "INTERACTS_WITH",  # drug-drug interactions
        "TREATS"           # drug treats diagnosis
    }

    # properties for each node type
    NODE_PROPERTIES: Dict[str, Set[str]] = {
        "Drug": {
            "name",           # drug name (string)
            "ndc",            # national drug code (string)
            "frequency"       # how common it is (string)
        },
        "Diagnosis": {
            "name",           # diagnosis name (string)
            "icd_code"        # icd-10 code (string)
        }
    }

    # properties for each relationship type
    RELATIONSHIP_PROPERTIES: Dict[str, Set[str]] = {
        "INTERACTS_WITH": {
            "severity",        # major, moderate, minor
            "effect",          # description of interaction effect
            "evidence_level",  # strength of evidence
            "mechanism"        # how the interaction occurs
        },
        "TREATS": {
            "indication",      # primary or secondary use
            "effectiveness"    # how well it works
        }
    }

    # cypher patterns that are explicitly allowed
    ALLOWED_PATTERNS: List[str] = [
        # find drugs that interact with a specific drug
        "(d1:Drug)-[i:INTERACTS_WITH]-(d2:Drug)",

        # find diagnoses treated by a drug
        "(d:Drug)-[t:TREATS]->(diag:Diagnosis)",

        # find drugs that treat a diagnosis
        "(d:Drug)-[t:TREATS]->(diag:Diagnosis)",
    ]

    # cypher operations that are forbidden (read-only queries only)
    FORBIDDEN_OPERATIONS: Set[str] = {
        "CREATE",
        "DELETE",
        "REMOVE",
        "MERGE",
        "SET",
        "DROP",
        "DETACH DELETE",
        "CALL",  # block procedures unless explicitly allowed
    }

    @classmethod
    def get_schema_description(cls) -> str:
        """
        generate human-readable schema description for llm prompt injection

        returns:
            formatted string describing the schema
        """
        description = """
Graph Schema:

Node Labels:
- Drug: Represents pharmaceutical drugs
  Properties: name (string), ndc (string), frequency (string)

- Diagnosis: Represents medical conditions
  Properties: name (string), icd_code (string)

Relationship Types:
- INTERACTS_WITH: Connects two Drug nodes (bidirectional)
  Properties: severity (major|moderate|minor), effect (string), evidence_level (string), mechanism (string)
  Pattern: (d1:Drug)-[i:INTERACTS_WITH]-(d2:Drug)

- TREATS: Connects Drug to Diagnosis
  Properties: indication (string), effectiveness (string)
  Pattern: (d:Drug)-[t:TREATS]->(diag:Diagnosis)

Important Rules:
1. All string comparisons must use toLower() for case-insensitive matching
2. Drug interactions are bidirectional - do not use directional arrows (-> or <-)
3. Only use properties that exist in the schema
4. Only use node labels and relationship types defined above
5. Generate read-only queries only (MATCH + RETURN)
"""
        return description.strip()

    @classmethod
    def get_example_queries(cls) -> List[Dict[str, str]]:
        """
        get example natural language to cypher pairs for few-shot prompting

        returns:
            list of dicts with 'question' and 'cypher' keys
        """
        return [
            {
                "question": "What drugs have major interactions with Warfarin?",
                "cypher": """MATCH (d1:Drug)-[i:INTERACTS_WITH]-(d2:Drug)
WHERE toLower(d1.name) = 'warfarin' AND toLower(i.severity) = 'major'
RETURN d2.name AS drug, i.effect AS effect, i.mechanism AS mechanism"""
            },
            {
                "question": "Show all interactions for Aspirin",
                "cypher": """MATCH (d1:Drug)-[i:INTERACTS_WITH]-(d2:Drug)
WHERE toLower(d1.name) = 'aspirin'
RETURN d2.name AS interacting_drug, i.severity AS severity, i.effect AS effect"""
            },
            {
                "question": "What diagnoses does Metformin treat?",
                "cypher": """MATCH (d:Drug)-[t:TREATS]->(diag:Diagnosis)
WHERE toLower(d.name) = 'metformin'
RETURN diag.name AS diagnosis, diag.icd_code AS code, t.effectiveness AS effectiveness"""
            },
            {
                "question": "Find drugs that treat Diabetes",
                "cypher": """MATCH (d:Drug)-[t:TREATS]->(diag:Diagnosis)
WHERE toLower(diag.name) = 'diabetes'
RETURN d.name AS drug, t.indication AS indication"""
            }
        ]

    @classmethod
    def validate_node_label(cls, label: str) -> bool:
        """check if node label exists in schema"""
        return label in cls.NODE_LABELS

    @classmethod
    def validate_relationship_type(cls, rel_type: str) -> bool:
        """check if relationship type exists in schema"""
        return rel_type in cls.RELATIONSHIP_TYPES

    @classmethod
    def validate_node_property(cls, label: str, property_name: str) -> bool:
        """check if property exists for given node label"""
        if label not in cls.NODE_PROPERTIES:
            return False
        return property_name in cls.NODE_PROPERTIES[label]

    @classmethod
    def validate_relationship_property(cls, rel_type: str, property_name: str) -> bool:
        """check if property exists for given relationship type"""
        if rel_type not in cls.RELATIONSHIP_PROPERTIES:
            return False
        return property_name in cls.RELATIONSHIP_PROPERTIES[rel_type]


# singleton instance for easy access
MOCK_SCHEMA = MockGraphSchema()
