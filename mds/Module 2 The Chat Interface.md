# Module 2: The Chat Interface
- status: done
- type: log
- id: implementation.phase1.ui.log
- owner: user
- last_checked: 2026-01-24T11:45:00+01:00
- context_dependencies: { "plan": "Phase 1 Plan.md", "app": "../src/app.py" }
<!-- content -->

## Overview
This module implements the user-facing layer of the Local Nexus. It uses **Streamlit** to provide a reactive, chat-based interface that feels like a modern AI tool but runs entirely locally.

## Architecture

### 1. Main Application (`src/app.py`)
- **Entry Point**: Sets up the page config ("Local Nexus", Wide Layout).
- **Robustness**: Includes global exception handling to display stack traces in the UI (essential for Cloud debugging).
- **Path Handling**: Includes robust `sys.path` injection to support various deployment environments (Local vs Cloud).

### 2. Sidebar (`src/components/sidebar.py`)
- **Data Management Hub**:
    - **File Uploader**: Supports CSV, Excel, JSON.
    - **Active Tables**: dynamically queries DuckDB to show what is currently loaded.
    - **Preview**: (Placeholder) Button to preview data.

### 3. Chat Component (`src/components/chat.py`)
- **Session State**: Manages the message history (`st.session_state.messages`).
- **Loopback Mock**:
    - Currently implements a "Mock Analyst".
    - Echoes user input ("Analyzed: ...").
    - If user types "show", it generates a dummy DataFrame to demonstrate UI rendering capabilities.
- **Cloud Detection**: Automatically detects if the app is running on Streamlit Cloud (`/mount/src/...`) or if the database is in-memory, and displays a generic warning banner.

## Usage
1.  **Launch**: `streamlit run src/app.py`.
2.  **Interact**: Type natural language queries in the chat box.
3.  **Visuals**: See DataFrames rendered natively in the chat stream.

## Verification
- [x] App loads without error.
- [x] Sidebar shows Uploader and Table List.
- [x] Chat input accepts text.
- [x] Mock response is generated.
- [x] "Show" keyword triggers a DataFrame render.
