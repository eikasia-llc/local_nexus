# Project Setup Guide
- status: active
- type: plan
<!-- content -->
> **Purpose:** Instructions for creating a simulation project with the required directory structure and components.

---

## Required Project Structure
- status: active
- type: task
<!-- content -->
Every simulation project must have **6 core parts minimum**:

```
project_name/
├── frontend/          # React/TypeScript UI (display layer only)
├── backend/           # Python/FastAPI simulation logic
├── database/          # Data storage (local JSON or remote DB)
├── AI_AGENTS/         # Documentation for AI assistants
├── tests/             # Unit tests (pytest)
└── notebooks/         # Jupyter notebooks for Google Colab
```

---

## 1. Frontend (`frontend/`)
- status: active
- type: task
<!-- content -->
**Purpose:** Display layer for human subjects. No simulation logic.

**Required files:**
```
frontend/
├── src/
│   ├── main.tsx          # React entry point
│   ├── App.tsx           # Root component (health check)
│   ├── App.css           # App styles
│   ├── Controls.tsx      # Main game UI component
│   ├── Controls.css      # Game UI styles
│   ├── index.css         # Global CSS variables
│   └── vite-env.d.ts     # Vite type declarations
├── index.html            # HTML entry point
├── package.json          # Node dependencies
├── vite.config.ts        # Vite configuration
└── tsconfig.json         # TypeScript configuration
```

**Guidelines:**
- Use Vite + React + TypeScript
- Premium dark theme with modern aesthetics
- All game logic via API calls to backend
- Display states received from backend

---

## 2. Backend (`backend/`)
- status: active
- type: task
<!-- content -->
**Purpose:** All simulation logic, agent behavior, and game state management.

**Required structure:**
```
backend/
├── api/
│   ├── __init__.py
│   ├── main.py           # FastAPI app with CORS
│   ├── routes.py         # API endpoints
│   └── session.py        # Session management
├── engine/
│   ├── __init__.py
│   ├── config.py         # SimulationConfig (Pydantic)
│   ├── state.py          # GameState models (Pydantic)
│   └── model.py          # Main simulation class
├── agents.py             # Agent implementations
├── environment.py        # Environment/game logic
├── logging.py            # Data export to JSON
└── __init__.py
```

**Guidelines:**
- Use FastAPI with CORS for localhost:5173
- Pydantic models for config and state
- All game logic in Python (not frontend)
- Auto-save session data on game completion

---

## 3. Database (`database/` or `data/`)
- status: active
- type: task
<!-- content -->
**Purpose:** Store session data, behavioral logs, and experimental results.

**Options:**

### Local Storage (Simple)
- status: active
- type: task
<!-- content -->
```
data/
└── sessions/
    └── {session_id}.json    # One file per game session
```

### Remote Database (Production)
- status: active
- type: task
<!-- content -->
- **MongoDB:** For document storage
- **Google Sheets:** For easy data sharing
- Configure via environment variables

**Required JSON format for sessions:**
```json
{
  "session_id": "uuid",
  "timestamp": "ISO-8601",
  "metadata": { "agent_name": "...", "num_rounds": 10 },
  "config": { ... },
  "final_scores": { "human": 25, "agent": 22 },
  "steps": [
    { "step": 0, "human_action": "...", "agent_action": "...",
      "outcome_human": 3, "outcome_agent": 3, "next_step": 1, "done": false }
  ]
}
```

---

## 4. AI_AGENTS (`AI_AGENTS/`)
- status: active
- type: task
<!-- content -->
**Purpose:** Documentation and instructions for AI coding assistants.

**Required files:**
```
AI_AGENTS/
├── REACT_ASSISTANT.md       # Guide for React + FastAPI setup
├── Student_Instructions.md  # Human-readable project guide
└── UI_DESIGN_ASSISTANT.md   # UI/UX design guidelines
```

**Guidelines:**
- Markdown format for AI context
- Include code examples and patterns
- Reference textbooks/guides for specialized techniques

---

## 5. Tests (`tests/`)
- status: active
- type: task
<!-- content -->
**Purpose:** Unit tests for backend logic using pytest.

**Required structure:**
```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures
├── test_environment.py      # Environment/payoff tests
├── test_agents.py           # Agent behavior tests
├── test_api.py              # API endpoint tests
└── test_engine.py           # Engine/model tests
```

**Guidelines:**
- Minimum coverage: environment + agents
- Run with: `python -m pytest tests/ -v`
- All tests must pass before deployment

---

## 6. Notebooks (`notebooks/`)
- status: active
- type: task
<!-- content -->
**Purpose:** Jupyter notebooks for running experiments in Google Colaboratory.

**Required structure:**
```
notebooks/
├── experiment_interface.ipynb    # Interactive experiment (Colab)
├── analysis_report.ipynb         # Results visualization
└── README.md                     # Instructions for Colab setup
```

**Guidelines for Google Colab:**
1. Add cell to install dependencies:
   ```python
   !pip install -q fastapi uvicorn pydantic
   ```

2. Add cell to clone repository:
   ```python
   !git clone https://github.com/username/repo.git
   %cd repo
   ```

3. Add path to import backend modules:
   ```python
   import sys
   sys.path.insert(0, '/content/repo')
   ```

4. Keep notebooks self-contained with clear markdown explanations

---

## Root Level Files
- status: active
- type: task
<!-- content -->
```
project_name/
├── AGENTS.md              # Main documentation for AI assistants
├── AGENTS_LOG.md          # Change log / intervention history
├── HOUSEKEEPING.md        # Testing protocol
├── requirements.txt       # Python dependencies
└── README.md              # Project overview
```

---

## Quick Setup Commands
- status: active
- type: task
<!-- content -->

### Create New Project
- status: active
- type: task
<!-- content -->
```bash
mkdir project_name && cd project_name
mkdir -p frontend backend/api backend/engine data/sessions AI_AGENTS tests notebooks
touch requirements.txt AGENTS.md AGENTS_LOG.md HOUSEKEEPING.md README.md
```

### Initialize Frontend
- status: active
- type: task
<!-- content -->
```bash
npm create vite@latest frontend -- --template react-ts
```

### Run Project
- status: active
- type: task
<!-- content -->
```bash

# Terminal 1: Backend
- status: active
- type: plan
<!-- content -->
python -m uvicorn backend.api.main:app --reload --port 8000

# Terminal 2: Frontend
- status: active
- type: plan
<!-- content -->
cd frontend && npm run dev
```

### Run Tests
- status: active
- type: task
<!-- content -->
```bash
python -m pytest tests/ -v
```

---

## Checklist for New Projects
- status: active
- type: task
<!-- content -->
- [ ] `frontend/` - Vite + React + TypeScript
- [ ] `backend/` - FastAPI with api/, engine/, agents.py, environment.py
- [ ] `data/sessions/` - JSON storage for session data
- [ ] `AI_AGENTS/` - Documentation for AI assistants
- [ ] `tests/` - pytest unit tests (min: environment + agents)
- [ ] `notebooks/` - Jupyter notebooks for Colab
- [ ] `requirements.txt` - Python dependencies
- [ ] `AGENTS.md` - Project description
- [ ] `HOUSEKEEPING.md` - Testing protocol
