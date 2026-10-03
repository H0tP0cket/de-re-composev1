# Code excerpts

Four files from the private research harness. They are here to show how recording, restoring, stall detection and arm construction work. They import modules that are not included, so they do not run on their own.

| File | What it does |
|---|---|
| `environment.py` | The Docker worker environment. It captures the workspace (W) as a tar with a fidelity manifest (type, mode, owner, size, mtime, SHA-256 per entry), captures the runtime (R) as a `docker diff` outside `/workspace` plus the process table, and restores both with exact manifest checks. A failed capture raises; nothing is ever marked branchable without validation. |
| `recorder.py` | Event emission. Every model request, response, tool call, tool result and delivered observation is stored with its exact bytes and a visibility class, so the history (H) the worker saw can be rebuilt byte for byte. |
| `stall_clock_v7.py` | The online stall detector (v7, v8, v8.1 rule versions). A stall is flagged at a valid, branchable graded checkpoint after at least N generated tokens and 5 turns without measurable progress, or (v8+) after a repeated rejection on the same requirement. The worker never sees it. |
| `treatments.py` | How an intervention arm is declared: which checkpoint donates each state channel (workspace, runtime, history), plus the checks that every donor exists, is branchable, belongs to the same run and comes before the stall (no future leaks). |
