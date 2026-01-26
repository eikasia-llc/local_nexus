# MCP Scikit-Learn Integration Plan
- status: active
- type: plan
- owner: antigravity
- context_dependencies: { "conventions": "../MD_CONVENTIONS.md" }
<!-- content -->

This document outlines the strategy for integrating **Scikit-Learn** into the Local Nexus via the **Model Context Protocol (MCP)**. This allows the chatbot to perform machine learning operations (training, prediction, data loading) in an isolated, standardized environment.

## Context & Objective
- status: active
- type: context
<!-- content -->
We aim to replicate the capabilities of `mcp-server-scikit-learn` to give our LLM tools to manipulate data.
*   **Goal**: Enable the Nexus Chatbot to classify/regress data stored in the Local Warehouse.
*   **Method**: Run a local MCP server that exposes Sklearn functions as **Tools**.
*   **Reference**: [shibuiwilliam/mcp-server-scikit-learn](https://github.com/shibuiwilliam/mcp-server-scikit-learn)

## Architecture
- status: active
- type: context
<!-- content -->
The system will follow a Client-Host-Server model.

1.  **Host (Local Nexus)**: The Streamlit app (`src/app.py`).
2.  **Client**: The Gemini LLM (via `google-generativeai`). *Note: Google's Native Client might not support MCP directly yet, so we may need a "Tooling Bridge" that translates Gemini Function Calls -> MCP Tool Calls.*
3.  **Server (Sklearn MCP)**: A standalone Python process running `fastmcp` or standard `mcp` SDK.

## Implementation Steps
- status: active
- type: plan
<!-- content -->

### Phase 1: Server Setup
1.  **Dependencies**:
    *   `mcp`
    *   `scikit-learn`
    *   `pandas`
    *   `numpy`
2.  **Server Script** (`src/mcp_server/sklearn_server.py`):
    *   Initialize `FastMCP("sklearn")`.
    *   Expose tools: `train_model`, `predict`, `evaluate_model`.

### Phase 2: Tool Definitions
Define the specific tools the LLM can call.

#### 1. Data Loading
*   `load_data(table_name)`: Fetch data from the local DuckDB warehouse.

#### 2. Training
*   `train_model(model_type, target_column, hyperparameters)`:
    *   Supports: `RandomForestClassifier`, `LinearRegression`, `LogisticRegression`.
    *   Returns: A unique `model_id` and metrics (Accuracy/R2).

#### 3. Inference
*   `predict(model_id, input_data)`: Use a trained model to make predictions.

### Phase 3: Client Integration
1.  **Bridge Layer** (`src/core/mcp_client.py`):
    *   Start the MCP server subprocess (`stdio`).
    *   Fetch tool definitions (`list_tools`).
    *   Convert MCP Tools -> Gemini `tools` format.
2.  **Chat Loop Update**:
    *   Pass tools to `model.generate_content`.
    *   Handle `function_call` responses by routing them to the MCP Client.

## Execution Checklist
- status: todo
- type: task
<!-- content -->

### Dependencies
- [ ] Add `mcp` to `requirements.txt`.
- [ ] Add `scikit-learn` to `requirements.txt`.

### Server Development
- [ ] Create `src/mcp_server/` directory.
- [ ] Implement `sklearn_server.py` using `FastMCP`.
- [ ] Implement `train_model` tool.
- [ ] Implement `predict` tool.

### Client Integration
- [ ] Create `src/core/mcp_bridge.py`.
- [ ] Wire up Gemini Function Calling to MCP Bridge.
