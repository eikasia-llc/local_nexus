# Self-Improvement & Update Protocol
- id: protocol.update
- status: active
- type: protocol
- context_dependencies: {"conventions": "MD_CONVENTIONS.md", "project_root": "README.md"}
- last_checked: 2026-01-25
<!-- content -->
This protocol defines the standard procedure for **learning from experience** and systematically updating the codebase, agents, and instructions. It transforms ad-hoc bug fixes into permanent system improvements.

## 1. Triggers
- status: active
- type: protocol
<!-- content -->
An update cycle is triggered by:
1.  **Execution Failure**: A script fails (e.g., `git clone` error, `ImportError`).
2.  **Logic Failure**: An agent follows instructions but achieves a suboptimal result (e.g., `apply_types.py` skipping files).
3.  **Ambiguity**: An agent is forced to ask the user for clarification because instructions were vague.
4.  **Inefficiency**: A process works but takes excessive steps or manual intervention.
5.  **Integration Failure**: CI/CD pipeline breaks, tests fail, or dependency conflicts arise (e.g., `pytest` failure).
6.  **Resource Exhaustion**: API rate limits, out of memory, or timeout errors.
7.  **Security Alert**: Dependency vulnerability or leaked secret detected.
8.  **Scalability Bottleneck**: Performance degrades with increased load (e.g., O(N^2) algorithm on large dataset).

## 2. The Learning Cycle (OODA Loop)
- status: active
- type: protocol
<!-- content -->
When a trigger occurs, the acting agent MUST execute the following loop *before* marking the task as complete.

### A. Observe
Capture the raw evidence of the failure or inefficiency.
- **Log it**: Record the exact error message, traceback, or confusing output.
- **Snapshot**: If applicable, save the state of the relevant file.

### B. Orient
Analyze the root cause. Ask:
- *Why did the code fail?* (e.g., "The filename pattern didn't match 'DEFINITIONS'.")
- *Why did I make that mistake?* (e.g., "The instructions in `CLEANER_AGENT.md` didn't specify checking that file.")
- *Is this a one-off or a systemic issue?*

### C. Decide
Determine the level of fix required:
1.  **Level 1 (Hotfix)**: Just fix the immediate bug to unblock. (Allowed only for low-priority/one-off issues).
2.  **Level 2 (Tool Update)**: Modify the tool script to handle this case permanently (e.g., adding heuristics to `apply_types.py`).
3.  **Level 3 (Instruction Update)**: Update the Agent's `.md` file to change their behavior or checklist.
4.  **Level 4 (Convention Update)**: Update `MD_CONVENTIONS.md` or `MASTER_PLAN.md` because the system design itself was flawed.

### D. Act
Execute the fix.
- Apply the code change.
- **CRITICAL**: Update the documentation/instructions immediately.

## 3. Codification (The "Latch")
- status: active
- type: protocol
<!-- content -->
Learning is only valid if it is **codified**—written down in a way that prevents the same error from happening to *any* instance of the agent in the future.

### Rules for Codification
1.  **Prefer Code over Text**: If you can write a script to enforce a rule (like `clean_repo.py`), do that instead of just writing "Please do X" in a markdown file.
2.  **Update the Source of Truth**:
    - If a tool needed parameters, update the **Agent's Tool Section**.
    - If a heuristic failed, update the **Script**.
    - If a dependency was missing, update the **Metadata**.
3.  **Log the Learning**: In `AGENTS_LOG.md` (or the specific agent's log), explicitly state what was learned:
    > "Updated `apply_types.py` to support content scanning because filename matching failed for `INFRASTRUCTURE_DEFINITIONS.md`."

## 4. Example Case Study
- status: active
- type: context
<!-- content -->
**Incident**: `apply_types.py` skipped `INFRASTRUCTURE_DEFINITIONS.md` because "DEFINITIONS" wasn't in its allowlist.
**Observation**: The file existed but had no metadata type.
**Orientation**: The script relied only on filenames. This is brittle.
**Decision**: Level 2 Update (Tool Update). Add "DEFINITIONS" to keywords AND implement content scanning fallback.
**Action**: Modified `apply_types.py`.
**Codification**: The script is now permanently smarter. Future "DEFINITIONS" files will be handled automatically.
