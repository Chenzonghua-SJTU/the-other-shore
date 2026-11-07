# Official pipeline development workspace

This repository is the `official_run` workspace. The initial commit preserves the BF2 Agent used by E10 R05/R06 and the ongoing R07–R11 noisy baseline. Upstream scientific source: `6f8a1599b604d4fd317176d8503c84c1fd636866`. BF2 runtime overlays change protocol feedback and actual prompts; this is not behaviorally identical to upstream.

## Boundaries and rollback

- Official SDK files outside `agent/` match `shore-contest-starter-v15.zip`; do not edit them. `.gitignore` is repository metadata added for version control.
- Implement tools, adapters, tests, experiment drivers and records inside `agent/`.
- Preserve original observations. Filtering may prepare symbolic-search data; parameter fitting, rollout validation and scoring use original observations and official `rift.rollout`.
- Never inject public-world identities, reference force laws or hidden evaluation data into the Agent.
- Tag the initial commit as `baseline-bf2-v1`; develop subsequent changes in separate commits. A frozen experiment directory is immutable.

## Baseline protocol

R07–R11 each run all six public worlds with observation noise, seeds 42–46 respectively; Qwen3.8-27B, 4096 output tokens/call, thinking disabled, PySR enabled, energy 10, exploration 300 seconds, model seed unset. Four independent model ports 8000–8003 are used. Each registered attempt is retained. Report five-score mean, sample standard deviation, range, valid submissions and unrounded score >=90 count per world. Infrastructure errors invalidate completeness and must be reported separately.

The active baseline remains in the previous workspace to avoid moving running processes:
`/inspire/hdd/project/26-the-other-shore/263140062/official_test/agent/_experiments/E10-20261107-224311`.

## Runtime and dependencies

Scientific source and configuration are versioned; model weights, virtual environments, Julia/PySR runtimes, transient service files and session credentials are not. The server currently uses Python 3.12.16 at `/opt/shore-test5-forge2/bin/python`, NumPy 2.1.3, SciPy 1.15.3 and SymPy 1.14.0. The server's ignored `agent/vendor_runtime` points to the existing runtime bundle. The initial PySR launcher still expects that layout; a fresh clone alone does not install these dependencies. Never commit that symlink, credentials or the large bundle.

Run protocol regression checks from repository root with the configured Python, `PYTHONDONTWRITEBYTECODE=1` and `-B -m agent.test_reliability_bf2` (also `test_background_bf2`, `test_bridge`, `test_selected_pair`). Create ignored `agent/_tests` first. Current test evidence and source hashes are in `agent/provenance/`.

`strategy.json` retains its historical BF1 label to preserve exact frozen-source identity; the actual BF2 overlay and hashes are recorded in provenance. Do not infer implementation version from that legacy label alone.
