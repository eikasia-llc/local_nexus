# Module 1: Local Data Warehouse
- status: done
- type: log
- id: implementation.phase1.warehouse.log
- owner: user
- last_checked: 2026-01-24T11:45:00+01:00
- context_dependencies: { "plan": "Phase 1 Plan.md", "ingestion": "../src/core/ingestion.py" }
<!-- content -->

## Overview
This module implements the "Long-Term Memory" of the Local Nexus system. It builds a persistent, optimized data storage layer on the user's local machine, ensuring privacy and zero-latency access.

## Architecture

### 1. Database Manager (`src/core/database.py`)
- **Technology**: DuckDB (Embedded OLAP).
- **Persistence**: connects to `data/warehouse.db`.
- **Cloud Compatibility**: Automatically falls back to `:memory:` (In-Memory Database) if the file system is read-only (e.g., Streamlit Cloud), ensuring the app never crashes.
- **Key Methods**:
    - `get_active_tables()`: Returns a list of all ingested files.
    - `register_file()`: Logs metadata (hash, filename, timestamp) to the internal `metadata_registry`.

### 2. Ingestion Service (`src/core/ingestion.py`)
- **Purpose**: Handles the "ETL" (Extract, Transform, Load) logic.
- **Capabilities**:
    - **CSV**: Parsed via Pandas.
    - **Excel**: Parsed via Pandas (requires `openpyxl`).
    - **JSON**: Parsed via Pandas (supports list-of-objects).
- **Features**:
    - **Deduplication**: Calculates SHA-256 hash of content. Prevents re-ingesting the same file.
    - **Sanitization**: Converts "My Column Name" -> `my_column_name` for SQL compatibility.

## Usage
1.  **Ingest**: Use the Sidebar "Upload" widget.
2.  **Verify**: Check "Active Tables" in the Sidebar.
3.  **Persist**: Restarting the app (Locally) preserves the data.

## Verification
- [x] Can upload CSV.
- [x] Can upload Excel.
- [x] Can upload JSON.
- [x] Data persists across restarts (Local).
- [x] Cloud Warning appears on Cloud Environment.
