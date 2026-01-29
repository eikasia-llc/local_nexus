"""
Unit tests for the Unified Engine module (Phase 4).

Tests query orchestration, retrieval integration, and end-to-end flow.
"""

import pytest
import duckdb
from unittest.mock import Mock, MagicMock
from src.core.unified_engine import (
    UnifiedEngine, RetrievalResult, EngineResponse
)


class TestRetrievalResultDataclass:
    """Tests for RetrievalResult dataclass."""

    def test_create_basic_result(self):
        """Test basic result creation."""
        result = RetrievalResult(
            source='vector',
            content='Test content'
        )
        assert result.source == 'vector'
        assert result.content == 'Test content'
        assert result.metadata == {}
        assert result.score == 0.0

    def test_create_full_result(self):
        """Test result with all fields."""
        result = RetrievalResult(
            source='sql',
            content='Query results',
            metadata={'sql': 'SELECT *'},
            score=0.95
        )
        assert result.metadata['sql'] == 'SELECT *'
        assert result.score == 0.95


class TestEngineResponseDataclass:
    """Tests for EngineResponse dataclass."""

    def test_success_response(self):
        """Test successful response creation."""
        response = EngineResponse(
            answer='The total is 100',
            query_type='structured',
            sql_query='SELECT COUNT(*) FROM table'
        )
        assert response.answer == 'The total is 100'
        assert response.query_type == 'structured'
        assert response.error is None

    def test_error_response(self):
        """Test error response creation."""
        response = EngineResponse(
            answer='',
            query_type='error',
            error='Database connection failed'
        )
        assert response.error == 'Database connection failed'


class TestUnifiedEngineInitialization:
    """Tests for UnifiedEngine initialization."""

    def test_init_minimal(self):
        """Test initialization with no components."""
        engine = UnifiedEngine()
        assert engine.vector_store is None
        assert engine.db_connection is None
        assert engine.llm_func is None

    def test_init_with_llm(self):
        """Test initialization with LLM function."""
        mock_llm = Mock(return_value="Test response")
        engine = UnifiedEngine(llm_func=mock_llm)
        assert engine.llm_func is mock_llm

    def test_lazy_router_initialization(self):
        """Router should be lazily initialized."""
        engine = UnifiedEngine()
        assert engine._router is None
        # Accessing router triggers initialization
        router = engine.router
        assert router is not None
        assert engine._router is not None


class TestQueryDecomposition:
    """Tests for query decomposition functionality."""

    @pytest.fixture
    def engine_with_mock_llm(self):
        """Engine with mock LLM for decomposition."""
        def mock_llm(prompt: str) -> str:
            if "Break down" in prompt:
                return "What are the total sales?\nHow many customers?"
            return "Mock response"

        return UnifiedEngine(llm_func=mock_llm, enable_decomposition=True)

    def test_decomposition_disabled(self):
        """When disabled, returns original query."""
        engine = UnifiedEngine(enable_decomposition=False)
        result = engine.decompose_query("Complex multi-part question")
        assert result == ("Complex multi-part question",)

    def test_decomposition_without_llm(self):
        """Without LLM, returns original query."""
        engine = UnifiedEngine(llm_func=None, enable_decomposition=True)
        result = engine.decompose_query("Complex question")
        assert result == ("Complex question",)

    def test_decomposition_with_llm(self, engine_with_mock_llm):
        """LLM decomposition should split queries."""
        result = engine_with_mock_llm.decompose_query("What are sales and customers?")
        assert len(result) == 2
        assert "total sales" in result[0].lower()

    def test_decomposition_caching(self, engine_with_mock_llm):
        """Decomposition results should be cached."""
        query = "Test query for caching"
        result1 = engine_with_mock_llm.decompose_query(query)
        result2 = engine_with_mock_llm.decompose_query(query)

        # Results should be identical (cached)
        assert result1 == result2

        # Check cache info
        cache_info = engine_with_mock_llm.decompose_query.cache_info()
        assert cache_info.hits >= 1

    def test_clear_cache(self, engine_with_mock_llm):
        """Cache should be clearable."""
        engine_with_mock_llm.decompose_query("Test query")
        cache_info_before = engine_with_mock_llm.decompose_query.cache_info()

        engine_with_mock_llm.clear_cache()

        cache_info_after = engine_with_mock_llm.decompose_query.cache_info()
        assert cache_info_after.hits == 0
        assert cache_info_after.misses == 0


class TestQueryRouting:
    """Tests for query type routing."""

    @pytest.fixture
    def engine(self):
        return UnifiedEngine()

    def test_structured_query_routing(self, engine):
        """Structured queries should be classified correctly."""
        query_type, _ = engine.retrieve("How many customers do we have?")
        assert query_type == 'structured'

    def test_unstructured_query_routing(self, engine):
        """Unstructured queries should be classified correctly."""
        query_type, _ = engine.retrieve("What is our refund policy?")
        assert query_type == 'unstructured'

    def test_force_type_override(self, engine):
        """Force type should override classification."""
        # This would normally be structured
        query_type, _ = engine.retrieve(
            "How many customers?",
            force_type='unstructured'
        )
        assert query_type == 'unstructured'


class TestVectorStoreRetrieval:
    """Tests for vector store retrieval integration."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        mock_vs = Mock()
        mock_vs.query.return_value = {
            'ids': [['doc1', 'doc2']],
            'documents': [['Document 1 content', 'Document 2 content']],
            'metadatas': [[{'source': 'test.md'}, {'source': 'other.md'}]],
            'distances': [[0.1, 0.3]]
        }
        return mock_vs

    def test_vector_retrieval(self, mock_vector_store):
        """Test retrieval from vector store."""
        engine = UnifiedEngine(vector_store=mock_vector_store)
        results = engine._retrieve_from_vector_store(['test query'])

        assert len(results) == 2
        assert results[0].source == 'vector'
        assert 'Document 1' in results[0].content
        assert results[0].score == 0.9  # 1.0 - 0.1 distance

    def test_vector_deduplication(self, mock_vector_store):
        """Duplicate documents should be deduplicated."""
        # Return same doc twice
        mock_vector_store.query.return_value = {
            'ids': [['doc1'], ['doc1']],
            'documents': [['Same content'], ['Same content']],
            'metadatas': [[{}], [{}]],
            'distances': [[0.1], [0.2]]
        }
        engine = UnifiedEngine(vector_store=mock_vector_store)
        results = engine._retrieve_from_vector_store(['query1', 'query2'])

        # Should only have one result (deduplicated)
        assert len(results) == 1

    def test_no_vector_store(self):
        """Should handle missing vector store gracefully."""
        engine = UnifiedEngine(vector_store=None)
        results = engine._retrieve_from_vector_store(['test'])
        assert results == []


class TestSQLRetrieval:
    """Tests for SQL retrieval integration."""

    @pytest.fixture
    def test_db(self):
        """Create test database."""
        conn = duckdb.connect(":memory:")
        conn.execute("""
            CREATE TABLE sales (
                id INTEGER,
                product VARCHAR,
                amount DECIMAL(10,2)
            )
        """)
        conn.execute("""
            INSERT INTO sales VALUES
            (1, 'Widget', 100.00),
            (2, 'Gadget', 200.00)
        """)
        yield conn
        conn.close()

    def test_sql_retrieval_with_mock_llm(self, test_db):
        """Test SQL retrieval with mock LLM."""
        def mock_llm(prompt: str) -> str:
            if "Generate" in prompt or "DuckDB" in prompt:
                return "SELECT product, SUM(amount) FROM sales GROUP BY product"
            return "The total sales are $300"

        engine = UnifiedEngine(db_connection=test_db, llm_func=mock_llm)
        results = engine._retrieve_from_sql("What are total sales by product?")

        assert len(results) == 1
        assert results[0].source == 'sql'
        assert 'SELECT' in results[0].metadata.get('sql', '')

    def test_sql_error_handling(self, test_db):
        """Test SQL error handling."""
        def mock_llm(prompt: str) -> str:
            return "SELECT * FROM nonexistent_table"

        engine = UnifiedEngine(db_connection=test_db, llm_func=mock_llm)
        results = engine._retrieve_from_sql("Bad query")

        assert len(results) == 1
        assert 'error' in results[0].metadata or 'Error' in results[0].content

    def test_no_db_connection(self):
        """Should handle missing DB connection gracefully."""
        engine = UnifiedEngine(db_connection=None)
        results = engine._retrieve_from_sql("test")
        assert results == []


class TestHybridRetrieval:
    """Tests for hybrid retrieval (combining vector + SQL)."""

    def test_hybrid_combines_sources(self):
        """Hybrid should combine both sources."""
        # Mock vector store
        mock_vs = Mock()
        mock_vs.query.return_value = {
            'ids': [['doc1']],
            'documents': [['Policy document']],
            'metadatas': [[{}]],
            'distances': [[0.1]]
        }

        # Mock DB
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE t (x INT)")
        conn.execute("INSERT INTO t VALUES (1)")

        def mock_llm(prompt: str) -> str:
            if "DuckDB" in prompt:
                return "SELECT COUNT(*) FROM t"
            return "Mock"

        engine = UnifiedEngine(
            vector_store=mock_vs,
            db_connection=conn,
            llm_func=mock_llm
        )

        query_type, results = engine.retrieve(
            "Test query",
            force_type='hybrid'
        )

        assert query_type == 'hybrid'
        sources = [r.source for r in results]
        assert 'vector' in sources
        assert 'sql' in sources

        conn.close()


class TestContextAssembly:
    """Tests for context assembly."""

    def test_assemble_empty_results(self):
        """Empty results should return default message."""
        engine = UnifiedEngine()
        context = engine._assemble_context([])
        assert "No relevant information" in context

    def test_assemble_single_result(self):
        """Single result should be formatted."""
        engine = UnifiedEngine()
        results = [RetrievalResult(source='vector', content='Test content')]
        context = engine._assemble_context(results)

        assert '[Source 1: vector]' in context
        assert 'Test content' in context

    def test_context_truncation(self):
        """Long content should be truncated."""
        engine = UnifiedEngine(max_context_tokens=50)  # Very small limit
        results = [RetrievalResult(
            source='vector',
            content='X' * 1000  # Long content
        )]
        context = engine._assemble_context(results)

        # Should be truncated
        assert len(context) < 1000


class TestEndToEndQuery:
    """End-to-end query tests."""

    @pytest.fixture
    def full_engine(self):
        """Create engine with mock components."""
        # Mock vector store
        mock_vs = Mock()
        mock_vs.query.return_value = {
            'ids': [['doc1']],
            'documents': [['Our refund policy allows returns within 30 days.']],
            'metadatas': [[{'source': 'policy.md'}]],
            'distances': [[0.1]]
        }
        mock_vs.get_stats.return_value = {'count': 10}

        # Mock DB
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE customers (id INT, name VARCHAR)")
        conn.execute("INSERT INTO customers VALUES (1, 'Alice'), (2, 'Bob')")

        def mock_llm(prompt: str) -> str:
            if "Break down" in prompt:
                return prompt.split('"')[1]  # Return original query
            if "DuckDB" in prompt:
                return "SELECT COUNT(*) FROM customers"
            return "Based on the context, there are 2 customers."

        return UnifiedEngine(
            vector_store=mock_vs,
            db_connection=conn,
            llm_func=mock_llm
        )

    def test_structured_query_flow(self, full_engine):
        """Test end-to-end structured query."""
        response = full_engine.query("How many customers do we have?")

        assert response.query_type == 'structured'
        assert response.error is None
        assert response.answer is not None
        assert response.sql_query is not None

    def test_unstructured_query_flow(self, full_engine):
        """Test end-to-end unstructured query."""
        response = full_engine.query("What is the refund policy?")

        assert response.query_type == 'unstructured'
        assert response.error is None
        assert len(response.sources) > 0

    def test_query_returns_sources(self, full_engine):
        """Sources should be included by default."""
        response = full_engine.query("What is the refund policy?")
        assert len(response.sources) > 0

    def test_query_without_sources(self, full_engine):
        """Sources can be excluded."""
        response = full_engine.query(
            "What is the refund policy?",
            return_sources=False
        )
        assert len(response.sources) == 0


class TestEngineStats:
    """Tests for engine statistics."""

    def test_get_stats_minimal(self):
        """Stats should work with minimal config."""
        engine = UnifiedEngine()
        stats = engine.get_stats()

        assert 'decomposition_enabled' in stats
        assert 'has_vector_store' in stats
        assert 'has_db_connection' in stats

    def test_get_stats_with_components(self):
        """Stats should include component info."""
        mock_vs = Mock()
        mock_vs.get_stats.return_value = {'count': 5}

        engine = UnifiedEngine(vector_store=mock_vs)
        stats = engine.get_stats()

        assert stats['has_vector_store'] is True
        assert 'vector_store' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
