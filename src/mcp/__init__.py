"""
MCP Server package for Local Nexus.

Exposes the Unified Nexus Architecture (RAG + Data Warehouse + Graph)
to LLM agents via the Model Context Protocol.

Tool Categories:
- Query Interface: unified_query, classify_query
- Structured Data: execute_sql, list_tables, describe_table
- Unstructured Data: semantic_search, list_document_sources
- Graph Data: find_connections, get_neighbors
- Ingestion: ingest_document, ingest_table
- System: get_system_status, clear_cache
"""

from src.mcp.server import create_server, run_server

__all__ = ['create_server', 'run_server']
