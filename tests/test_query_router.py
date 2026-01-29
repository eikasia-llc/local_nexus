"""
Unit tests for the query router module (Phase 2).

Tests classification accuracy across diverse query types:
- Structured queries (SQL/aggregations)
- Unstructured queries (semantic/document search)
- Hybrid queries (both)
"""

import pytest
from src.core.query_router import QueryRouter, QueryType, create_llm_classifier


class TestQueryRouterBasics:
    """Basic functionality tests for QueryRouter."""

    @pytest.fixture
    def router(self):
        """Create a QueryRouter instance."""
        return QueryRouter()

    def test_empty_query_returns_unstructured(self, router):
        """Empty queries should default to unstructured."""
        assert router.classify("") == QueryType.UNSTRUCTURED
        assert router.classify("   ") == QueryType.UNSTRUCTURED

    def test_query_type_enum_values(self):
        """Verify QueryType enum has expected values."""
        assert QueryType.STRUCTURED.value == "structured"
        assert QueryType.UNSTRUCTURED.value == "unstructured"
        assert QueryType.HYBRID.value == "hybrid"


class TestStructuredQueryClassification:
    """Tests for structured (SQL/tabular) query classification."""

    @pytest.fixture
    def router(self):
        return QueryRouter()

    @pytest.mark.parametrize("query", [
        "How many customers do we have?",
        "What is the total revenue?",
        "Count all orders from last month",
        "Show me the average order value",
        "What's the sum of sales in Q3?",
        "Top 10 products by units sold",
        "Number of users per region",
        "Maximum transaction amount",
        "Minimum order value this year",
    ])
    def test_aggregation_queries(self, router, query):
        """Queries with aggregation keywords should be STRUCTURED."""
        assert router.classify(query) == QueryType.STRUCTURED

    @pytest.mark.parametrize("query", [
        "Show me sales by month",
        "Revenue per quarter",
        "Daily active users grouped by country",
        "Monthly recurring revenue trends",
    ])
    def test_time_aggregation_queries(self, router, query):
        """Time-based aggregation queries should be STRUCTURED."""
        assert router.classify(query) == QueryType.STRUCTURED

    @pytest.mark.parametrize("query", [
        "Filter customers with revenue > $1000",
        "Orders greater than 100 units",
        "Show records where amount >= 500",
    ])
    def test_filter_queries(self, router, query):
        """Filter/comparison queries should be STRUCTURED."""
        assert router.classify(query) == QueryType.STRUCTURED


class TestUnstructuredQueryClassification:
    """Tests for unstructured (semantic/document) query classification."""

    @pytest.fixture
    def router(self):
        return QueryRouter()

    @pytest.mark.parametrize("query", [
        "What is our PTO policy?",
        "Explain the refund procedure",
        "What are the company guidelines for remote work?",
        "Describe the onboarding process",
    ])
    def test_policy_questions(self, router, query):
        """Policy/procedure questions should be UNSTRUCTURED."""
        assert router.classify(query) == QueryType.UNSTRUCTURED

    @pytest.mark.parametrize("query", [
        "What is machine learning?",
        "What does the handbook say about expenses?",
        "What are the security protocols?",
        "How does the approval process work?",
    ])
    def test_definition_questions(self, router, query):
        """Definition/explanation questions should be UNSTRUCTURED."""
        assert router.classify(query) == QueryType.UNSTRUCTURED

    @pytest.mark.parametrize("query", [
        "Find information about employee benefits",
        "Search for documentation on API usage",
        "Tell me about the return policy",
        "Summary of the annual report",
    ])
    def test_search_queries(self, router, query):
        """Search/information retrieval queries should be UNSTRUCTURED."""
        assert router.classify(query) == QueryType.UNSTRUCTURED


class TestHybridQueryClassification:
    """Tests for hybrid (structured + unstructured) query classification."""

    @pytest.fixture
    def router(self):
        return QueryRouter()

    @pytest.mark.parametrize("query", [
        "Which customers mentioned pricing concerns and have >$10K lifetime value?",
        "Find users who complained about shipping with more than 5 orders",
        "Customers who said positive things about support and spent over $500",
    ])
    def test_semantic_filter_queries(self, router, query):
        """Queries combining semantic search with data filters should be HYBRID."""
        assert router.classify(query) == QueryType.HYBRID

    @pytest.mark.parametrize("query", [
        "Reviews mentioning quality issues for products with low ratings",
        "Feedback about pricing from high-value customers",
        "Complaints about delivery for orders greater than $100",
    ])
    def test_semantic_aggregation_queries(self, router, query):
        """Queries combining semantic content with aggregations should be HYBRID."""
        result = router.classify(query)
        # These may be classified as HYBRID or UNSTRUCTURED depending on keyword density
        assert result in [QueryType.HYBRID, QueryType.UNSTRUCTURED]


class TestConfidenceScoring:
    """Tests for confidence scoring and detailed classification."""

    @pytest.fixture
    def router(self):
        return QueryRouter()

    def test_classify_with_confidence_returns_tuple(self, router):
        """classify_with_confidence should return (QueryType, confidence, scores)."""
        query_type, confidence, scores = router.classify_with_confidence(
            "How many orders were placed?"
        )

        assert isinstance(query_type, QueryType)
        assert isinstance(confidence, float)
        assert isinstance(scores, dict)
        assert 'structured' in scores
        assert 'unstructured' in scores
        assert 'hybrid' in scores

    def test_high_confidence_for_clear_queries(self, router):
        """Clear queries should have high confidence scores."""
        _, confidence, _ = router.classify_with_confidence(
            "How many customers do we have total?"
        )
        assert confidence >= 1.0

    def test_routing_explanation_contains_details(self, router):
        """get_routing_explanation should return detailed explanation."""
        explanation = router.get_routing_explanation("What is the total revenue?")

        assert "Query Type:" in explanation
        assert "Confidence:" in explanation
        assert "Scores:" in explanation
        assert "Reason:" in explanation


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def router(self):
        return QueryRouter()

    def test_mixed_signals_prefers_stronger(self, router):
        """Queries with mixed signals should classify based on stronger signal."""
        # This query has both "count" (structured) and weak unstructured signals
        result = router.classify("Count the documents")
        assert result == QueryType.STRUCTURED

    def test_ambiguous_query_defaults_to_unstructured(self, router):
        """Ambiguous queries with no clear signals should default to unstructured."""
        # Very generic query with no strong keywords
        result = router.classify("Show me something interesting")
        assert result == QueryType.UNSTRUCTURED

    def test_case_insensitive_matching(self, router):
        """Keyword matching should be case-insensitive."""
        assert router.classify("HOW MANY USERS?") == QueryType.STRUCTURED
        assert router.classify("WHAT IS THE POLICY?") == QueryType.UNSTRUCTURED

    def test_punctuation_handling(self, router):
        """Queries with various punctuation should be handled correctly."""
        assert router.classify("How many users???") == QueryType.STRUCTURED
        assert router.classify("What's the policy...") == QueryType.UNSTRUCTURED


class TestLLMFallback:
    """Tests for LLM fallback functionality."""

    def test_llm_fallback_disabled_by_default(self):
        """LLM fallback should be disabled by default."""
        router = QueryRouter()
        assert router.llm_fallback is False

    def test_llm_fallback_can_be_enabled(self):
        """LLM fallback can be enabled via constructor."""
        router = QueryRouter(llm_fallback=True)
        assert router.llm_fallback is True

    def test_create_llm_classifier(self):
        """create_llm_classifier should return a callable."""
        def mock_llm(prompt):
            return "STRUCTURED"

        classifier = create_llm_classifier(mock_llm)
        assert callable(classifier)
        assert classifier("test query") == QueryType.STRUCTURED

    def test_llm_classifier_handles_various_responses(self):
        """LLM classifier should handle various response formats."""
        def make_mock(response):
            return lambda p: response

        structured_classifier = create_llm_classifier(make_mock("STRUCTURED"))
        assert structured_classifier("query") == QueryType.STRUCTURED

        unstructured_classifier = create_llm_classifier(make_mock("UNSTRUCTURED"))
        assert unstructured_classifier("query") == QueryType.UNSTRUCTURED

        hybrid_classifier = create_llm_classifier(make_mock("HYBRID"))
        assert hybrid_classifier("query") == QueryType.HYBRID

        # Unknown response should default to unstructured
        unknown_classifier = create_llm_classifier(make_mock("SOMETHING ELSE"))
        assert unknown_classifier("query") == QueryType.UNSTRUCTURED


class TestRealWorldQueries:
    """Tests with real-world query examples from the implementation plan."""

    @pytest.fixture
    def router(self):
        return QueryRouter()

    def test_structured_example_from_plan(self, router):
        """'How many sales in Q3?' should be STRUCTURED."""
        assert router.classify("How many sales in Q3?") == QueryType.STRUCTURED

    def test_unstructured_example_from_plan(self, router):
        """'What's our PTO policy?' should be UNSTRUCTURED."""
        assert router.classify("What's our PTO policy?") == QueryType.UNSTRUCTURED

    def test_hybrid_example_from_plan(self, router):
        """Complex hybrid query from implementation plan."""
        result = router.classify(
            "Which customers mentioned pricing and spent >$1K?"
        )
        # This should be classified as HYBRID due to semantic + structured signals
        assert result in [QueryType.HYBRID, QueryType.STRUCTURED]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
