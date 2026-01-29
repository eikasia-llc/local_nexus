"""
Unit tests for the Text2SQL module (Phase 3).

Tests schema introspection, SQL validation, and query execution.
Uses in-memory DuckDB for isolation.
"""

import pytest
import duckdb
from src.core.text2sql import (
    Text2SQL, SchemaIntrospector, TableSchema, SQLResult
)


class TestSchemaIntrospector:
    """Tests for SchemaIntrospector class."""

    @pytest.fixture
    def test_db(self):
        """Create an in-memory test database with sample tables."""
        conn = duckdb.connect(":memory:")

        # Create sample tables
        conn.execute("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name VARCHAR,
                email VARCHAR,
                created_at TIMESTAMP
            )
        """)
        conn.execute("""
            INSERT INTO customers VALUES
            (1, 'Alice', 'alice@example.com', '2024-01-15 10:00:00'),
            (2, 'Bob', 'bob@example.com', '2024-01-20 14:30:00'),
            (3, 'Charlie', 'charlie@example.com', '2024-02-01 09:15:00')
        """)

        conn.execute("""
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                amount DECIMAL(10,2),
                order_date DATE
            )
        """)
        conn.execute("""
            INSERT INTO orders VALUES
            (101, 1, 150.00, '2024-01-16'),
            (102, 1, 200.00, '2024-01-25'),
            (103, 2, 350.00, '2024-02-01')
        """)

        # Create a system table that should be excluded
        conn.execute("""
            CREATE TABLE metadata_registry (
                file_id VARCHAR PRIMARY KEY,
                filename VARCHAR
            )
        """)

        yield conn
        conn.close()

    @pytest.fixture
    def introspector(self, test_db):
        """Create a SchemaIntrospector instance."""
        return SchemaIntrospector(test_db)

    def test_get_all_tables_excludes_system_tables(self, introspector):
        """System tables should be excluded from results."""
        tables = introspector.get_all_tables()

        assert 'customers' in tables
        assert 'orders' in tables
        assert 'metadata_registry' not in tables

    def test_get_table_schema_returns_correct_columns(self, introspector):
        """Table schema should include all columns with correct types."""
        schema = introspector.get_table_schema('customers')

        assert schema is not None
        assert schema.name == 'customers'
        assert len(schema.columns) == 4

        column_names = [c['name'] for c in schema.columns]
        assert 'id' in column_names
        assert 'name' in column_names
        assert 'email' in column_names

    def test_get_table_schema_returns_row_count(self, introspector):
        """Table schema should include accurate row count."""
        schema = introspector.get_table_schema('customers')

        assert schema.row_count == 3

    def test_get_table_schema_returns_sample_data(self, introspector):
        """Table schema should include sample data."""
        schema = introspector.get_table_schema('customers', sample_rows=2)

        assert len(schema.sample_data) == 2
        assert 'name' in schema.sample_data[0]

    def test_get_table_schema_nonexistent_table(self, introspector):
        """Nonexistent table should return None."""
        schema = introspector.get_table_schema('nonexistent_table')
        assert schema is None

    def test_get_full_schema_context_includes_all_tables(self, introspector):
        """Schema context should describe all user tables."""
        context = introspector.get_full_schema_context()

        assert 'customers' in context
        assert 'orders' in context
        assert 'metadata_registry' not in context
        assert 'Columns:' in context

    def test_get_full_schema_context_includes_sample_data(self, introspector):
        """Schema context should include sample data."""
        context = introspector.get_full_schema_context()

        assert 'Sample data:' in context
        assert 'Row 1:' in context


class TestText2SQLValidation:
    """Tests for SQL validation in Text2SQL."""

    @pytest.fixture
    def test_db(self):
        """Create a minimal test database."""
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE test_table (id INTEGER, value VARCHAR)")
        yield conn
        conn.close()

    @pytest.fixture
    def text2sql(self, test_db):
        """Create a Text2SQL instance without LLM."""
        return Text2SQL(test_db, llm_func=None, read_only=True)

    def test_validate_empty_query(self, text2sql):
        """Empty queries should be invalid."""
        is_valid, error = text2sql._validate_sql("")
        assert is_valid is False
        assert "Empty" in error

    def test_validate_select_query(self, text2sql):
        """Valid SELECT queries should pass validation."""
        is_valid, error = text2sql._validate_sql("SELECT * FROM test_table")
        assert is_valid is True
        assert error is None

    def test_validate_blocks_delete(self, text2sql):
        """DELETE queries should be blocked in read-only mode."""
        is_valid, error = text2sql._validate_sql("DELETE FROM test_table")
        assert is_valid is False
        assert "delete" in error.lower()

    def test_validate_blocks_drop(self, text2sql):
        """DROP queries should be blocked in read-only mode."""
        is_valid, error = text2sql._validate_sql("DROP TABLE test_table")
        assert is_valid is False
        assert "drop" in error.lower()

    def test_validate_blocks_insert(self, text2sql):
        """INSERT queries should be blocked in read-only mode."""
        is_valid, error = text2sql._validate_sql("INSERT INTO test_table VALUES (1, 'x')")
        assert is_valid is False
        assert "insert" in error.lower()

    def test_validate_blocks_update(self, text2sql):
        """UPDATE queries should be blocked in read-only mode."""
        is_valid, error = text2sql._validate_sql("UPDATE test_table SET value = 'x'")
        assert is_valid is False
        assert "update" in error.lower()

    def test_validate_syntax_error(self, text2sql):
        """Syntax errors should be caught."""
        # Use a SELECT query with syntax error (missing FROM)
        is_valid, error = text2sql._validate_sql("SELECT * FORM test_table")
        assert is_valid is False
        # Should catch the syntax error via EXPLAIN
        assert error is not None

    def test_write_allowed_when_not_readonly(self, test_db):
        """Write operations allowed when read_only=False."""
        text2sql = Text2SQL(test_db, llm_func=None, read_only=False)
        # Note: This will still fail syntax validation via EXPLAIN
        # but shouldn't fail the keyword check
        is_valid, error = text2sql._validate_sql("INSERT INTO test_table VALUES (1, 'x')")
        # The validation should pass keyword check but may fail on execution
        # For this test, we're checking it doesn't block on keywords
        assert "not allowed" not in (error or "")


class TestSQLExtraction:
    """Tests for SQL extraction from LLM responses."""

    @pytest.fixture
    def text2sql(self):
        """Create a Text2SQL instance."""
        conn = duckdb.connect(":memory:")
        return Text2SQL(conn)

    def test_extract_plain_sql(self, text2sql):
        """Extract SQL from plain response."""
        response = "SELECT * FROM users"
        result = text2sql._extract_sql(response)
        assert result == "SELECT * FROM users"

    def test_extract_sql_with_markdown(self, text2sql):
        """Extract SQL from markdown code block."""
        response = """Here's the query:
```sql
SELECT name, email FROM users WHERE active = true
```
"""
        result = text2sql._extract_sql(response)
        assert "SELECT name, email FROM users" in result

    def test_extract_sql_with_generic_markdown(self, text2sql):
        """Extract SQL from generic markdown code block."""
        response = """```
SELECT COUNT(*) FROM orders
```"""
        result = text2sql._extract_sql(response)
        assert "SELECT COUNT(*) FROM orders" in result

    def test_extract_sql_removes_trailing_semicolon(self, text2sql):
        """Trailing semicolons should be removed."""
        response = "SELECT * FROM users;"
        result = text2sql._extract_sql(response)
        assert not result.endswith(';')

    def test_extract_sql_handles_whitespace(self, text2sql):
        """Whitespace should be trimmed."""
        response = "   SELECT * FROM users   \n\n"
        result = text2sql._extract_sql(response)
        assert result == "SELECT * FROM users"


class TestText2SQLExecution:
    """Tests for SQL execution."""

    @pytest.fixture
    def test_db(self):
        """Create test database with data."""
        conn = duckdb.connect(":memory:")
        conn.execute("""
            CREATE TABLE products (
                id INTEGER,
                name VARCHAR,
                price DECIMAL(10,2),
                category VARCHAR
            )
        """)
        conn.execute("""
            INSERT INTO products VALUES
            (1, 'Laptop', 999.99, 'Electronics'),
            (2, 'Mouse', 29.99, 'Electronics'),
            (3, 'Desk', 199.99, 'Furniture'),
            (4, 'Chair', 149.99, 'Furniture')
        """)
        yield conn
        conn.close()

    @pytest.fixture
    def text2sql(self, test_db):
        return Text2SQL(test_db)

    def test_execute_simple_select(self, text2sql):
        """Execute a simple SELECT query."""
        result = text2sql.execute_sql("SELECT * FROM products")

        assert result.success is True
        assert result.row_count == 4
        assert 'name' in result.columns

    def test_execute_with_aggregation(self, text2sql):
        """Execute query with aggregation."""
        result = text2sql.execute_sql(
            "SELECT category, COUNT(*) as count FROM products GROUP BY category"
        )

        assert result.success is True
        assert result.row_count == 2
        assert 'category' in result.columns
        assert 'count' in result.columns

    def test_execute_with_filter(self, text2sql):
        """Execute query with WHERE clause."""
        result = text2sql.execute_sql(
            "SELECT name FROM products WHERE price > 100"
        )

        assert result.success is True
        assert result.row_count == 3  # Laptop, Desk, Chair

    def test_execute_returns_error_for_invalid_sql(self, text2sql):
        """Invalid SQL should return error result."""
        result = text2sql.execute_sql("SELECT * FROM nonexistent_table")

        assert result.success is False
        assert result.error is not None

    def test_execute_returns_columns(self, text2sql):
        """Result should include column names."""
        result = text2sql.execute_sql("SELECT name, price FROM products LIMIT 1")

        assert result.columns == ['name', 'price']


class TestText2SQLEndToEnd:
    """End-to-end tests with mock LLM."""

    @pytest.fixture
    def test_db(self):
        """Create test database."""
        conn = duckdb.connect(":memory:")
        conn.execute("""
            CREATE TABLE sales (
                id INTEGER,
                product VARCHAR,
                amount DECIMAL(10,2),
                sale_date DATE
            )
        """)
        conn.execute("""
            INSERT INTO sales VALUES
            (1, 'Widget A', 100.00, '2024-01-15'),
            (2, 'Widget B', 250.00, '2024-01-20'),
            (3, 'Widget A', 150.00, '2024-02-01'),
            (4, 'Widget B', 300.00, '2024-02-15')
        """)
        yield conn
        conn.close()

    def test_query_with_mock_llm(self, test_db):
        """Test full query flow with mock LLM."""
        # Mock LLM that returns a valid SQL query
        def mock_llm(prompt: str) -> str:
            return "SELECT product, SUM(amount) as total FROM sales GROUP BY product"

        text2sql = Text2SQL(test_db, llm_func=mock_llm)
        result = text2sql.query("What are the total sales by product?")

        assert result.success is True
        assert result.row_count == 2
        assert result.sql is not None

    def test_query_with_markdown_response(self, test_db):
        """Test handling of markdown-formatted LLM response."""
        def mock_llm(prompt: str) -> str:
            return """Here's the SQL query:
```sql
SELECT COUNT(*) as total_sales FROM sales
```
"""

        text2sql = Text2SQL(test_db, llm_func=mock_llm)
        result = text2sql.query("How many sales do we have?")

        assert result.success is True
        assert result.row_count == 1

    def test_query_without_llm_returns_error(self, test_db):
        """Query without LLM configured should return error."""
        text2sql = Text2SQL(test_db, llm_func=None)
        result = text2sql.query("How many sales?")

        assert result.success is False
        assert "No LLM" in result.error

    def test_generate_sql_only(self, test_db):
        """Test generating SQL without executing."""
        def mock_llm(prompt: str) -> str:
            return "SELECT * FROM sales WHERE amount > 200"

        text2sql = Text2SQL(test_db, llm_func=mock_llm)
        sql, error = text2sql.generate_sql("Show expensive sales")

        assert error is None
        assert "SELECT" in sql
        assert "amount > 200" in sql

    def test_get_available_tables(self, test_db):
        """Test listing available tables."""
        text2sql = Text2SQL(test_db)
        tables = text2sql.get_available_tables()

        assert 'sales' in tables

    def test_get_table_info(self, test_db):
        """Test getting table information."""
        text2sql = Text2SQL(test_db)
        info = text2sql.get_table_info('sales')

        assert info is not None
        assert info.name == 'sales'
        assert info.row_count == 4


class TestSQLResultDataclass:
    """Tests for SQLResult dataclass."""

    def test_success_result(self):
        """Test successful result creation."""
        result = SQLResult(
            success=True,
            sql="SELECT 1",
            data=[(1,)],
            columns=['value'],
            row_count=1
        )

        assert result.success is True
        assert result.error is None

    def test_error_result(self):
        """Test error result creation."""
        result = SQLResult(
            success=False,
            error="Table not found"
        )

        assert result.success is False
        assert result.error == "Table not found"
        assert result.data is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
