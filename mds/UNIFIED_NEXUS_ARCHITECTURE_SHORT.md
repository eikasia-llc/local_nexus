# Unified Nexus Architecture (Summary)
- status: active
- type: context
- id: unified-nexus-short
- last_checked: 2026-02-02
<!-- content -->
A condensed reference for the **Unified RAG + Data Warehouse** architecture. For full implementation details, see `UNIFIED_NEXUS_ARCHITECTURE.md` and `KG_UNIFIED_NEXUS_ARCHITECTURE.md`.

## Core Insight
- status: active
- type: context
<!-- content -->
RAG and Data Warehouse are **complementary**:

| Capability | RAG (Unstructured) | Data Warehouse (Structured) |
|:-----------|:-------------------|:----------------------------|
| **Data Type** | Documents, PDFs | Tables, CSVs |
| **Query Style** | Semantic similarity | SQL computation |
| **Strengths** | Fuzzy matching | Aggregations, joins |
| **Example** | "What's our PTO policy?" | "Total sales in Q3?" |

**Hybrid questions require BOTH**: *"Which customers complained about shipping and spent >$1K?"*

This is the **TAG (Table-Augmented Generation)** paradigm.

## Architecture Overview
- status: active
- type: context
<!-- content -->

```
┌─────────────────────────────────────┐
│           User Question             │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│         Query Router (LLM)          │
│  Classifies: STRUCTURED/UNSTRUCTURED│
│              /HYBRID                │
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│ChromaDB│ │ DuckDB │ │ Graph  │
│  RAG   │ │Text2SQL│ │ Store  │
└───┬────┘ └───┬────┘ └───┬────┘
    └──────────┼──────────┘
               ▼
┌─────────────────────────────────────┐
│     Context Assembly + LLM Gen      │
└─────────────────────────────────────┘
```

## Component Summary
- status: active
- type: context
<!-- content -->

| Component | Technology | Purpose |
|:----------|:-----------|:--------|
| **Vector Store** | ChromaDB | Semantic search over documents |
| **Data Warehouse** | DuckDB | SQL queries over structured data |
| **Query Router** | LLM + heuristics | Classify query type |
| **Text2SQL** | LLM + schema | Natural language → SQL |
| **Graph Store** | JSON/DuckDB | Relationship traversal |
| **MCP Server** | 12 tools | Programmatic agent access |

## Key Features
- status: active
- type: context
<!-- content -->

### Smart Retrieval (Query Decomposition)
Complex questions are broken into sub-queries for better recall:
- *"Top products and shipping feedback?"* → 2 sub-queries (SQL + RAG)
- LRU cache prevents redundant LLM calls

### Batch Vector Search
**82% latency reduction** via parallel ChromaDB queries:
```python
# Single batch request for all sub-queries
results = vector_store.query(query_texts=decomposed_queries, n_results=top_k)
```

### MCP Tools (12 Available)
| Tool | Category | Purpose |
|:-----|:---------|:--------|
| `unified_query` | Query | Auto-routed answers |
| `execute_sql` | Structured | Direct SQL execution |
| `semantic_search` | Unstructured | Document search |
| `find_connections` | Graph | Path finding |
| `get_neighbors` | Graph | Direct connections |

## File Structure
- status: active
- type: context
<!-- content -->

```
src/core/
├── unified_engine.py      # Main orchestrator
├── query_router.py        # STRUCTURED/UNSTRUCTURED/HYBRID
├── vector_store.py        # ChromaDB wrapper
├── text2sql.py            # NL → SQL generation
├── graph_store.py         # Relationship traversal
├── document_ingestion.py  # TXT/MD/PDF/DOCX → vectors
└── database.py            # DuckDB connection

src/mcp/
└── server.py              # 12 agent tools
```

## Quick Usage
- status: active
- type: context
<!-- content -->

```python
from src.core.unified_engine import UnifiedEngine

engine = UnifiedEngine(
    db_path="data/warehouse.db",
    vector_store_path="data/vectordb",
    llm_client=llm
)

# Auto-routes to appropriate path
result = engine.query("Which customers mentioned pricing and spent >$1K?")
print(result['answer'])
```

## Graph Integration (Optional)
- status: active
- type: context
<!-- content -->

| Approach | Complexity | Best For |
|:---------|:-----------|:---------|
| Text Serialization | Low | Simple LLM prompts |
| Markdown Tables | Low | Agent protocols |
| DuckDB Graph Tables | Medium | SQL-integrated ⭐ |
| NetworkX | Medium | Network analysis |

**Recommended**: Start with text serialization, upgrade to DuckDB graph tables if traversal becomes complex.

## Error Handling
- status: active
- type: protocol
<!-- content -->

> [!WARNING]
> **Architectural Constraints**:
> - Lazy loading for heavy components
> - Initialization retries (no permanent `None` state)
> - Component errors trapped at boundary
