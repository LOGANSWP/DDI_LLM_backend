"""
unit tests for cypher validator

tests both safety checks and schema validation.
includes positive cases (valid queries) and negative cases (invalid/dangerous queries).
"""

import pytest
from src.validator import CypherValidator, QuerySanitizer, ValidationResult


class TestCypherValidatorSafety:
    """test safety checks for dangerous operations"""

    def setup_method(self):
        """create validator instance for each test"""
        self.validator = CypherValidator()

    def test_valid_read_query(self):
        """valid read-only query should pass"""
        query = """
        MATCH (d1:Drug)-[i:INTERACTS_WITH]-(d2:Drug)
        WHERE toLower(d1.name) = 'warfarin'
        RETURN d2.name, i.severity
        """
        result = self.validator.validate(query)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_reject_create_operation(self):
        """queries with CREATE should be rejected"""
        query = "CREATE (d:Drug {name: 'Test'})"
        result = self.validator.validate(query)
        assert result.is_valid is False
        assert any('CREATE' in err for err in result.errors)

    def test_reject_delete_operation(self):
        """queries with DELETE should be rejected"""
        query = "MATCH (d:Drug) DELETE d"
        result = self.validator.validate(query)
        assert result.is_valid is False
        assert any('DELETE' in err for err in result.errors)

    def test_reject_merge_operation(self):
        """queries with MERGE should be rejected"""
        query = "MERGE (d:Drug {name: 'Test'})"
        result = self.validator.validate(query)
        assert result.is_valid is False
        assert any('MERGE' in err for err in result.errors)

    def test_reject_set_operation(self):
        """queries with SET should be rejected"""
        query = "MATCH (d:Drug) SET d.name = 'Changed'"
        result = self.validator.validate(query)
        assert result.is_valid is False
        assert any('SET' in err for err in result.errors)

    def test_reject_drop_operation(self):
        """queries with DROP should be rejected"""
        query = "DROP INDEX ON :Drug(name)"
        result = self.validator.validate(query)
        assert result.is_valid is False
        assert any('DROP' in err for err in result.errors)

    def test_reject_detach_delete(self):
        """queries with DETACH DELETE should be rejected"""
        query = "MATCH (d:Drug) DETACH DELETE d"
        result = self.validator.validate(query)
        assert result.is_valid is False
        assert any('DETACH DELETE' in err or 'DELETE' in err for err in result.errors)

    def test_reject_call_procedure(self):
        """queries with CALL should be rejected"""
        query = "CALL db.schema.visualization()"
        result = self.validator.validate(query)
        assert result.is_valid is False
        assert any('CALL' in err for err in result.errors)

    def test_require_match_clause(self):
        """query must contain MATCH"""
        query = "RETURN 1"
        result = self.validator.validate(query)
        assert result.is_valid is False
        assert any('MATCH' in err for err in result.errors)

    def test_require_return_clause(self):
        """query must contain RETURN or WITH"""
        query = "MATCH (d:Drug)"
        result = self.validator.validate(query)
        assert result.is_valid is False
        assert any('RETURN' in err or 'WITH' in err for err in result.errors)

    def test_reject_query_chaining(self):
        """detect query chaining attempts"""
        query = "MATCH (d:Drug) RETURN d; MATCH (x) CREATE (x)"
        result = self.validator.validate(query)
        assert result.is_valid is False
        # should catch either injection or CREATE
        assert len(result.errors) > 0

    def test_empty_query(self):
        """empty query should fail validation"""
        result = self.validator.validate("")
        assert result.is_valid is False
        assert any('empty' in err.lower() for err in result.errors)

    def test_whitespace_only_query(self):
        """whitespace-only query should fail"""
        result = self.validator.validate("   \n\t  ")
        assert result.is_valid is False


class TestCypherValidatorSchema:
    """test schema validation checks"""

    def setup_method(self):
        """create validator instance for each test"""
        self.validator = CypherValidator()

    def test_valid_drug_label(self):
        """Drug label should be valid"""
        query = "MATCH (d:Drug) RETURN d.name"
        result = self.validator.validate(query)
        assert result.is_valid is True

    def test_valid_diagnosis_label(self):
        """Diagnosis label should be valid"""
        query = "MATCH (d:Diagnosis) RETURN d.name"
        result = self.validator.validate(query)
        assert result.is_valid is True

    def test_invalid_node_label(self):
        """unknown node label should fail"""
        query = "MATCH (p:Patient) RETURN p.name"
        result = self.validator.validate(query)
        assert result.is_valid is False
        assert any('Patient' in err for err in result.errors)

    def test_valid_interacts_with_relationship(self):
        """INTERACTS_WITH relationship should be valid"""
        query = "MATCH (d1:Drug)-[i:INTERACTS_WITH]-(d2:Drug) RETURN d1.name"
        result = self.validator.validate(query)
        assert result.is_valid is True

    def test_valid_treats_relationship(self):
        """TREATS relationship should be valid"""
        query = "MATCH (d:Drug)-[t:TREATS]->(diag:Diagnosis) RETURN d.name"
        result = self.validator.validate(query)
        assert result.is_valid is True

    def test_invalid_relationship_type(self):
        """unknown relationship type should fail"""
        query = "MATCH (d1:Drug)-[p:PRESCRIBED_BY]->(d2:Doctor) RETURN d1"
        result = self.validator.validate(query)
        assert result.is_valid is False
        assert any('PRESCRIBED_BY' in err for err in result.errors)

    def test_valid_drug_properties(self):
        """valid Drug properties should not generate errors"""
        query = "MATCH (d:Drug) RETURN d.name, d.ndc, d.frequency"
        result = self.validator.validate(query)
        assert result.is_valid is True

    def test_valid_diagnosis_properties(self):
        """valid Diagnosis properties should not generate errors"""
        query = "MATCH (d:Diagnosis) RETURN d.name, d.icd_code"
        result = self.validator.validate(query)
        assert result.is_valid is True

    def test_invalid_property_warning(self):
        """invalid property should generate warning"""
        query = "MATCH (d:Drug) RETURN d.name, d.invalid_property"
        result = self.validator.validate(query)
        # should still be valid but with warnings
        assert len(result.warnings) > 0
        assert any('invalid_property' in warn for warn in result.warnings)

    def test_case_sensitivity_warning(self):
        """query without toLower should generate warning"""
        query = "MATCH (d:Drug) WHERE d.name = 'Warfarin' RETURN d"
        result = self.validator.validate(query)
        # should be valid but warn about case sensitivity
        assert len(result.warnings) > 0
        assert any('case' in warn.lower() for warn in result.warnings)

    def test_case_insensitive_query_no_warning(self):
        """query with toLower should not warn"""
        query = "MATCH (d:Drug) WHERE toLower(d.name) = 'warfarin' RETURN d"
        result = self.validator.validate(query)
        # should have no case sensitivity warnings
        case_warnings = [w for w in result.warnings if 'case' in w.lower()]
        assert len(case_warnings) == 0


class TestCypherValidatorComplexQueries:
    """test validation of complex realistic queries"""

    def setup_method(self):
        self.validator = CypherValidator()

    def test_major_interactions_query(self):
        """validate query for major drug interactions"""
        query = """
        MATCH (d1:Drug)-[i:INTERACTS_WITH]-(d2:Drug)
        WHERE toLower(d1.name) = 'warfarin' AND toLower(i.severity) = 'major'
        RETURN d2.name AS drug, i.effect AS effect, i.mechanism AS mechanism
        """
        result = self.validator.validate(query)
        assert result.is_valid is True
        assert result.sanitized_query is not None

    def test_treats_diagnosis_query(self):
        """validate query for drugs treating a diagnosis"""
        query = """
        MATCH (d:Drug)-[t:TREATS]->(diag:Diagnosis)
        WHERE toLower(diag.name) = 'diabetes'
        RETURN d.name AS drug, t.indication AS indication
        """
        result = self.validator.validate(query)
        assert result.is_valid is True

    def test_multiple_conditions_query(self):
        """validate query with multiple WHERE conditions"""
        query = """
        MATCH (d1:Drug)-[i:INTERACTS_WITH]-(d2:Drug)
        WHERE toLower(d1.name) = 'aspirin'
        AND toLower(i.severity) IN ['major', 'moderate']
        RETURN d2.name, i.severity, i.effect
        ORDER BY i.severity
        """
        result = self.validator.validate(query)
        assert result.is_valid is True

    def test_count_aggregation_query(self):
        """validate query with aggregation"""
        query = """
        MATCH (d:Drug)-[i:INTERACTS_WITH]-(other:Drug)
        WHERE toLower(i.severity) = 'major'
        RETURN d.name, COUNT(other) AS interaction_count
        ORDER BY interaction_count DESC
        """
        result = self.validator.validate(query)
        assert result.is_valid is True


class TestQuerySanitizer:
    """test query sanitization utilities"""

    def test_normalize_whitespace(self):
        """normalize multiple spaces and newlines"""
        query = "MATCH  (d:Drug)  \n\n  RETURN   d.name"
        normalized = QuerySanitizer.normalize_whitespace(query)
        assert "  " not in normalized
        assert "\n" not in normalized
        assert normalized == "MATCH (d:Drug) RETURN d.name"

    def test_remove_line_comments(self):
        """remove single-line comments"""
        query = """
        MATCH (d:Drug) // this is a comment
        RETURN d.name
        """
        cleaned = QuerySanitizer.remove_comments(query)
        assert "//" not in cleaned
        assert "this is a comment" not in cleaned

    def test_remove_block_comments(self):
        """remove block comments"""
        query = """
        MATCH (d:Drug) /* this is a
        multi-line comment */
        RETURN d.name
        """
        cleaned = QuerySanitizer.remove_comments(query)
        assert "/*" not in cleaned
        assert "multi-line comment" not in cleaned

    def test_preserve_query_after_comment_removal(self):
        """ensure query still works after removing comments"""
        query = "MATCH (d:Drug) // comment\nRETURN d.name"
        cleaned = QuerySanitizer.remove_comments(query)
        assert "MATCH" in cleaned
        assert "RETURN" in cleaned
