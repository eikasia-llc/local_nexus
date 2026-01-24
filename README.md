# Central Planner
- status: active
- type: guideline
- context_dependencies: { "conventions": "MD_CONVENTIONS.md", "agents": "AGENTS.md" }
<!-- content -->

## Project Overview
- status: active
<!-- content -->
This repository serves as a **meta-project** to organize and coordinate multiple different repositories and workflows using AI agents. It implements a **Markdown-JSON Hybrid Schema** to define detailed implementation plans that are readable by both humans and AI agents, and losslessly convertible to JSON for programmatic manipulation.

## The Markdown-JSON Protocol
- status: active
<!-- content -->
The project uses a strict **Markdown-JSON Hybrid Schema** to treat documentation as a computable graph. Every Markdown file following this schema can be losslessly converted to JSON and back.

### Structure & Hierarchy
- status: active
<!-- content -->
- **Hierarchy**: Defined by standard Markdown headers (`#`, `##`, `###`). The nesting level indicates parent-child relationships, allowing for infinite depth.
- **Dependencies**:
    - **Implicit**: A child node is strictly dependent on its parent.
    - **Explicit (DAG)**: Defined in metadata using `blocked_by` (e.g., `blocked_by: [task-a, task-b]`). This allows defining complex Directed Acyclic Graphs where a node depends on multiple independent predecessors.
- **Identification**: 
    - **Title**: Default identification method.
    - **Global ID**: An optional `id` metadata field (e.g., `id: component.backend`). This is the **preferred** method for robust cross-file operations (merging/linking), as it persists even if the human-readable title changes.
- **Separator**: A `<!-- content -->` line separates metadata from content, ensuring unambiguous parsing.
- **Metadata**: Every node (header) must be immediately followed by a METADATA block defining its attributes.

**Example Node:**
```markdown

## Implement Login Feature
- id: feature.auth.login
- status: in-progress
- owner: @frontend-agent
- priority: high
- blocked_by: [feature.auth.backend, infra.db.setup]
<!-- content -->
Description of the login feature implementation...
```

### The Language System (Tooling)
- status: active
<!-- content -->
A suite of Python scripts in the `language/` directory allows agents and humans to interact with this protocol programmatically:

- **Parsing**: `md_parser.py` converts raw Markdown into structured Python objects, validating schema compliance.
- **Visualization**: 
    - `visualization.py`: Renders the hierarchical task tree in the terminal.
    - `visualize_dag.py`: Visualizes task dependencies (DAG) either as a **text report** (`--format text`) or **Mermaid syntax** (`--format mermaid`).
- **Standardization**: `migrate.py` automatically converts standard Markdown files into the Protocol format by injecting default metadata.
- **Import**: `importer.py` converts legacy documents (`.docx`, `.pdf`) into Protocol-compliant Markdown.
- **Operations**: `operations.py` allows for complex tree manipulations, such as merging subtrees or extending plans.
- **Maintenance**:
    - `remove_meta.py`: Strips metadata from files for clean reading or export.
    - `cli_utils.py`: Shared utilities for CLI argument parsing and file handling.
- **Testing**: `test.py` validates the functionality of the language tools.

## Repository Manager
- status: active
<!-- content -->
The `manager/` directory contains tools for coordinating multiple external repositories:

- **Cleaning**: `clean_repo.py` clones an external repository, extracts relevant Markdown files, and standardizes them using `migrate.py`. It supports standard Git URLs and branch URLs (e.g., `/tree/branch-name`).
- **Integration**: `update_master_plan.py` integrates these external plans into the central `MASTER_PLAN.md`.

## Directory Structure
- status: active
<!-- content -->
```
central_planner/
├── AGENTS.md                     # General Agent Guidelines & Workflow
├── README.md                     # Project Documentation (This File)
├── AI_AGENTS/                    # Agent Instructions
│   ├── specialists/              # Specialized Agent Protocols (e.g., React, RecSys)
│   └── generalists/              # General Project Protocols (Setup, Housekeeping)
├── manager/                      # Repository Management Tools
│   ├── MANAGER_PROTOCOL.md       # Manager Agent Protocol & Responsibilities
│   ├── MASTER_PLAN.md            # Aggregated plan from all repositories
│   ├── clean_repo.py             # Harvests and cleans external repos
│   ├── update_master_plan.py     # Integrates external plans
│   └── repolist.txt              # List of target repositories
├── language/                     # Tooling for the Communication System
│   ├── md_parser.py              # Parses & Validates MD/METADATA files
│   ├── visualization.py          # Visualizes task trees
│   ├── visualize_dag.py          # Visualizes task DAGs (Text/Mermaid)
│   ├── operations.py             # Merges and Extends task trees
│   ├── migrate.py                # Heuristic migration tool
│   ├── importer.py               # Imports legacy docs (.docx, .pdf)
│   ├── remove_meta.py            # Strips metadata
│   ├── cli_utils.py              # CLI Utilities
│   └── test.py                   # Test suite for the language tools
└── MD_CONVENTIONS.md # Specification of the Schema
```

## Key Files
- status: active
<!-- content -->
- **`MD_CONVENTIONS.md`**: The definitive guide on how to write files in this repository.
- **`AGENTS.md`**: The entry point for any AI agent joining the project.
