"""
cypher query validator and safety checker

validates generated cypher queries against schema and security rules.
ensures only safe, read-only queries are executed.
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.mock_schema import MOCK_SCHEMA


@dataclass
class ValidationResult:
    """result of cypher validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_query: Optional[str] = None


class CypherValidator:
    """
    validates cypher queries against schema and safety rules

    performs two types of validation:
    1. safety checks - prevents dangerous operations
    2. schema checks - ensures query uses valid labels/properties
    """

    def __init__(self, schema=MOCK_SCHEMA):
        self.schema = schema

    def validate(self, cypher_query: str) -> ValidationResult:
        """
        validate a cypher query

        args:
            cypher_query: the cypher query to validate

        returns:
            ValidationResult with validation status and errors
        """
        errors = []
        warnings = []

        # strip whitespace and normalize
        cypher_query = cypher_query.strip()

        if not cypher_query:
            errors.append("empty query")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # safety checks first
        safety_errors = self._check_safety(cypher_query)
        errors.extend(safety_errors)

        # schema validation
        schema_errors, schema_warnings = self._check_schema(cypher_query)
        errors.extend(schema_errors)
        warnings.extend(schema_warnings)

        # if valid, return sanitized query
        is_valid = len(errors) == 0
        sanitized = cypher_query if is_valid else None

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            sanitized_query=sanitized
        )

    def _check_safety(self, query: str) -> List[str]:
        """
        check for dangerous operations in query

        only read-only queries allowed (MATCH + RETURN/WITH)
        """
        errors = []
        query_upper = query.upper()

        # check for forbidden operations
        for forbidden_op in self.schema.FORBIDDEN_OPERATIONS:
            # use word boundaries to avoid false positives
            pattern = r'\b' + re.escape(forbidden_op) + r'\b'
            if re.search(pattern, query_upper):
                errors.append(f"forbidden operation detected: {forbidden_op}")

        # ensure query contains MATCH (required for read operations)
        if 'MATCH' not in query_upper:
            errors.append("query must contain MATCH clause")

        # ensure query contains RETURN (required to return results)
        if 'RETURN' not in query_upper and 'WITH' not in query_upper:
            errors.append("query must contain RETURN or WITH clause")

        # check for command injection attempts
        suspicious_patterns = [
            r';\s*MATCH',  # query chaining
            r'//.*CREATE',  # commented dangerous ops
            r'/\*.*CREATE.*\*/',  # block comment tricks
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, query, re.IGNORECASE | re.DOTALL):
                errors.append("suspicious pattern detected - possible injection attempt")

        return errors

    def _check_schema(self, query: str) -> Tuple[List[str], List[str]]:
        """
        validate query against schema definitions

        checks:
        - node labels exist in schema
        - relationship types exist in schema
        - properties exist for their respective labels/relationships

        returns:
            tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        # extract node labels
        node_labels = self._extract_node_labels(query)
        for label in node_labels:
            if not self.schema.validate_node_label(label):
                errors.append(f"unknown node label: {label}")

        # extract relationship types
        rel_types = self._extract_relationship_types(query)
        for rel_type in rel_types:
            if not self.schema.validate_relationship_type(rel_type):
                errors.append(f"unknown relationship type: {rel_type}")

        # extract properties and validate them
        # this is a simplified check - looks for pattern like "label.property"
        property_refs = self._extract_property_references(query)
        for var_name, prop_name in property_refs:
            # try to find which label this variable has
            label = self._find_label_for_variable(query, var_name, node_labels)
            if label:
                if not self.schema.validate_node_property(label, prop_name):
                    # could be a relationship property
                    rel_type = self._find_rel_type_for_variable(query, var_name, rel_types)
                    if rel_type:
                        if not self.schema.validate_relationship_property(rel_type, prop_name):
                            warnings.append(f"property '{prop_name}' not in schema for {rel_type}")
                    else:
                        warnings.append(f"property '{prop_name}' not in schema for {label}")

        # check for case sensitivity issues
        if not self._uses_case_insensitive_matching(query):
            warnings.append("query may have case sensitivity issues - consider using toLower()")

        return errors, warnings

    def _extract_node_labels(self, query: str) -> List[str]:
        """extract node labels from query like (:Drug) or (d:Drug)"""
        pattern = r'\([a-zA-Z0-9_]*:([a-zA-Z_][a-zA-Z0-9_]*)\)'
        matches = re.findall(pattern, query)
        return list(set(matches))

    def _extract_relationship_types(self, query: str) -> List[str]:
        """extract relationship types like [:INTERACTS_WITH] or [i:INTERACTS_WITH]"""
        pattern = r'\[[a-zA-Z0-9_]*:([A-Z_][A-Z0-9_]*)\]'
        matches = re.findall(pattern, query)
        return list(set(matches))

    def _extract_property_references(self, query: str) -> List[Tuple[str, str]]:
        """extract property references like d1.name or i.severity"""
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(pattern, query)
        return matches

    def _find_label_for_variable(self, query: str, var_name: str, known_labels: List[str]) -> Optional[str]:
        """try to find which label a variable is bound to"""
        # look for pattern like (var:Label)
        pattern = r'\(' + re.escape(var_name) + r':([a-zA-Z_][a-zA-Z0-9_]*)\)'
        match = re.search(pattern, query)
        if match:
            return match.group(1)
        return None

    def _find_rel_type_for_variable(self, query: str, var_name: str, known_rel_types: List[str]) -> Optional[str]:
        """try to find which relationship type a variable is bound to"""
        # look for pattern like [var:REL_TYPE]
        pattern = r'\[' + re.escape(var_name) + r':([A-Z_][A-Z0-9_]*)\]'
        match = re.search(pattern, query)
        if match:
            return match.group(1)
        return None

    def _uses_case_insensitive_matching(self, query: str) -> bool:
        """check if query uses toLower() for string comparisons"""
        # check for toLower usage in WHERE clauses
        return 'toLower(' in query or 'toUpper(' in query


class QuerySanitizer:
    """additional utilities for cleaning and normalizing queries"""

    @staticmethod
    def normalize_whitespace(query: str) -> str:
        """normalize whitespace in query"""
        # replace multiple spaces/newlines with single space
        normalized = re.sub(r'\s+', ' ', query)
        return normalized.strip()

    @staticmethod
    def remove_comments(query: str) -> str:
        """remove cypher comments from query"""
        # remove line comments
        query = re.sub(r'//.*$', '', query, flags=re.MULTILINE)
        # remove block comments
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
        return query.strip()
