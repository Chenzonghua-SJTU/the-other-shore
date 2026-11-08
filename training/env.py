"""Three-interaction-round, structured-action environment using the public console/scorer.

This is a local Python API, not a Gymnasium Env or an execution sandbox. A
policy receives noisy observations only. Scoring runs after a valid submission
closes the episode; reference arrays and truth labels never enter its result.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from development_worlds import catalog, evaluation
from development_worlds.console import DevelopmentConsole
from rift.law import load_law, validate_source
from scripts.validate_claims import validate_claims


MAX_INTERACTION_ROUNDS = 3
MAX_ACTIONS_PER_ROUND = 64
# Compatibility aliases; these have always counted calls to step(), not LLM calls.
MAX_TURNS = MAX_INTERACTION_ROUNDS
MAX_ACTIONS_PER_TURN = MAX_ACTIONS_PER_ROUND


def _json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _seed(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(value)


class DiscoveryEnv:
    """A reset/step interface for environment interaction batches and RL rewards.

    ``clock_mode='realtime'`` preserves the 300-second exploration deadline.
    Explicit ``clock_mode='frozen'`` disables elapsed-time expiry for reproducible
    training/data generation; energy, observations and three interaction rounds still apply.
    Batched actions execute in order, with errors included in the observation.
    Invalid actions consume an attempted interaction round, not experimental energy.
    Internal model calls, reasoning, fitting and local tools do not call step()
    and do not consume interaction rounds.

    Candidate law files are executed locally through the existing loader. Only
    use trusted local candidate code, or isolate the whole worker externally.
    """

    def __init__(self, worlds=None, *, clock_mode="realtime", anonymize_world_id=True):
        selected = catalog.WORLDS if worlds is None else worlds
        if isinstance(selected, (str, bytes)):
            selected = [selected]
        self.worlds = tuple(str(world) for world in selected)
        if not self.worlds or len(set(self.worlds)) != len(self.worlds):
            raise ValueError("worlds must be a nonempty sequence without duplicates")
        if any(world not in catalog.WORLDS for world in self.worlds):
            raise ValueError("worlds contains an unknown world ID")
        if clock_mode not in ("realtime", "frozen"):
            raise ValueError("clock_mode must be realtime or frozen")
        if type(anonymize_world_id) is not bool:
            raise ValueError("anonymize_world_id must be a boolean")
        self.clock_mode = clock_mode
        self.anonymize_world_id = anonymize_world_id
        self._rng = np.random.default_rng()
        self._console = None
        self._references = {}
        self._terminated = False
        self._truncated = False
        self._turn = 0
        self._reason = None
        self._terminal_budget = None

    @property
    def trainer_metadata(self):
        """Trainer-only bookkeeping; do not concatenate into the policy prompt."""
        if self._console is None:
            raise RuntimeError("Call reset first")
        return {"world_id": self._world, "observation_seed": self._observation_seed,
                "clock_mode": self.clock_mode, "max_turns": MAX_TURNS,
                "max_interaction_rounds": MAX_INTERACTION_ROUNDS}

    def _policy_value(self, value):
        value = _json_copy(value)
        if not self.anonymize_world_id:
            return value

        def walk(item):
            if isinstance(item, dict):
                return {key: ("training" if key == "world_id" else walk(child))
                        for key, child in item.items()}
            if isinstance(item, list):
                return [walk(child) for child in item]
            return item
        return walk(value)

    def _budget(self):
        return copy.deepcopy(self._terminal_budget if self._terminal_budget is not None
                             else self._console.budget)

    def _observation(self, results):
        return self._policy_value({"results": results,
                                   "interaction_round": self._turn,
                                   "interaction_rounds_remaining": MAX_INTERACTION_ROUNDS - self._turn,
                                   "turn": self._turn,
                                   "turns_remaining": MAX_TURNS - self._turn,
                                   "budget": self._budget()})

    def _info(self):
        return {"clock_mode": self.clock_mode, "reason": self._reason,
                "max_turns": MAX_TURNS, "max_interaction_rounds": MAX_INTERACTION_ROUNDS,
                "bootstrap_allowed": not (self._terminated or self._truncated)}

    def reset(self, *, seed=None, options=None):
        """Return ``(observation, info)``; initial mission/first light cost no interaction round.

        ``options={'world': '7'}`` selects a member of ``worlds``. Otherwise
        a seeded uniform draw selects the next world; reset without a new seed
        advances the existing RNG. Seed controls both task draw and noise.
        """
        if seed is not None:
            self._rng = np.random.default_rng(_seed(seed))
        if options is None:
            options = {}
        if not isinstance(options, dict) or set(options) - {"world"}:
            raise ValueError("reset options allows only world")
        if "world" in options:
            world = str(options["world"])
            if world not in self.worlds:
                raise ValueError("Requested world is not in this environment's world set")
        else:
            world = self.worlds[int(self._rng.integers(len(self.worlds)))]
        observation_seed = int(self._rng.integers(0, 2**63 - 1))
        clock = time.monotonic if self.clock_mode == "realtime" else lambda: 0.0
        console = DevelopmentConsole(world, catalog.noise_sigma(world), observation_seed,
                                     catalog.config(world), clock=clock)
        mission = console.mission()
        self._console = console
        self._world = world
        self._observation_seed = observation_seed
        self._turn = 0
        self._terminated = self._truncated = False
        self._reason = None
        self._terminal_budget = None
        result = {"method": "mission", "ok": True, "result": mission}
        return self._observation([result]), self._info()

    def _reference(self):
        # Track the same source/config/case identity enforced by evaluate_law.
        # Cache only clean references, never rewards or a candidate's result.
        identity = {"source": evaluation._source_hashes(),
                    "config": catalog.config(self._world),
                    "cases": catalog.validation_cases(self._world)}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cached = self._references.get(self._world)
        if cached is None or cached[0] != digest:
            reference = evaluation.calibrate_world(self._world)
            self._references[self._world] = (digest, reference)
        return self._references[self._world][1]

    def _score_submission(self, law_path, claims, energy_used):
        return evaluation.evaluate_law(self._world, law_path, claims, energy_used,
                                       calibration=self._reference())

    def _prepare_submission(self, params, directory):
        if set(params) != {"law_path", "claims"}:
            raise ValueError("submit requires exactly law_path and claims")
        if not isinstance(params["law_path"], str) or not params["law_path"]:
            raise ValueError("law_path must be a nonempty local file path")
        claims = validate_claims(params["claims"])
        path = Path(params["law_path"])
        # Snapshot the exact submitted bytes before accepting the artifact.
        with path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if not raw or len(raw) > 1024 * 1024:
            raise ValueError("law.py must contain 1 to 1048576 bytes")
        validate_source(raw.decode("utf-8-sig"))
        target = Path(directory) / "law.py"
        target.write_bytes(raw)
        load_law(target)  # Reject malformed PARAMS/API before closing the episode.
        return target, claims

    def _dispatch(self, method, params):
        if method in ("mission", "archive"):
            if params:
                raise ValueError(f"{method} accepts no parameters")
            return getattr(self._console, method)()
        if method == "observe":
            if set(params) != {"times"}:
                raise ValueError("observe requires exactly times")
            return self._console.observe(params["times"])
        if method in ("launch", "insert_probe"):
            if set(params) != {"probes", "t_launch", "readout", "source_properties"}:
                raise ValueError(f"{method} requires probes, t_launch, readout, source_properties")
            return self._console.launch(**params)
        raise ValueError("Unknown method; use mission, observe, insert_probe (alias launch), archive or submit")

    def step(self, actions):
        """Execute one environment interaction round containing one or many tools.

        Accept a request, an ordered request list, or explicit
        ``{'method': 'batch', 'params': {'actions': [requests...]}}``.
        ``insert_probe`` is an exact alias for ``launch`` (same inputs/cost).
        Local reasoning/model/tool calls outside step() do not count. A valid
        submit is terminal and ignores later requests in the same batch.
        Submit in interaction round three is accepted before applying the limit. Intermediate rewards
        and missing-submission terminal rewards are zero. Interaction-round/deadline limits
        are intrinsic task failures: no value bootstrap is allowed. Scoring happens
        after the exploration clock is stopped, as in the standard runner.
        """
        if self._console is None:
            raise RuntimeError("Call reset before step")
        if self._terminated or self._truncated:
            raise RuntimeError("Episode is over; call reset before another step")
        self._turn += 1
        results = []
        pending = None
        reward = 0.0
        with tempfile.TemporaryDirectory(prefix="shore-training-submit-") as directory:
            try:
                requests = _json_copy(actions)
                if isinstance(requests, dict) and requests.get("method") == "batch":
                    if (set(requests) != {"method", "params"}
                            or not isinstance(requests["params"], dict)
                            or set(requests["params"]) != {"actions"}):
                        raise ValueError("batch requires exactly params.actions containing request objects")
                    requests = requests["params"]["actions"]
                elif isinstance(requests, dict):
                    requests = [requests]
                if not isinstance(requests, list) or not 1 <= len(requests) <= MAX_ACTIONS_PER_ROUND:
                    raise ValueError(f"An interaction round requires 1 to {MAX_ACTIONS_PER_ROUND} requests")
            except (ValueError, TypeError, OverflowError) as exc:
                requests = []
                results.append({"method": None, "ok": False,
                                "error": {"type": type(exc).__name__, "message": str(exc)[:500]}})
            for request in requests:
                method = request.get("method") if isinstance(request, dict) else None
                try:
                    if (not isinstance(request, dict) or set(request) != {"method", "params"}
                            or not isinstance(method, str) or not isinstance(request["params"], dict)):
                        raise ValueError("Each request requires a string method and an object params")
                    # Also gate submit: it must not bypass the console deadline.
                    self._console._ensure_active()
                    params = request["params"]
                    if method == "submit":
                        pending = self._prepare_submission(params, directory)
                        self._console._ensure_active()
                        result = {"accepted": True, "submitted": True}
                        self._terminated = True
                        self._reason = "submitted"
                        self._terminal_budget = self._console.budget
                    else:
                        result = self._dispatch(method, params)
                    results.append({"method": method, "ok": True, "result": result})
                except MemoryError:
                    raise
                except Exception as exc:
                    results.append({"method": method, "ok": False,
                                    "error": {"type": type(exc).__name__, "message": str(exc)[:500]}})
                    if isinstance(exc, TimeoutError):
                        self._terminated = True
                        self._reason = "deadline"
                if self._terminated or self._truncated:
                    break
            if not (self._terminated or self._truncated):
                if self._console.budget["wall_clock_remaining_seconds"] <= 0:
                    self._terminated, self._reason = True, "deadline"
                elif self._turn >= MAX_INTERACTION_ROUNDS:
                    self._terminated, self._reason = True, "turn_limit"
            if (self._terminated or self._truncated) and self._terminal_budget is None:
                self._terminal_budget = self._console.budget
            info = self._info()
            if self._reason == "submitted" and pending is not None:
                # Infrastructure/reference failures propagate to the trainer;
                # they are never silently converted into scientific zero scores.
                law_path, claims = pending
                score = self._score_submission(law_path, claims, self._terminal_budget["energy_used"])
                reward = float(score["total"]) / 100.0
                # Whitelist scalar terminal diagnostics. No label/trajectory leak.
                info["score"] = {key: score[key] for key in
                                 ("prediction", "law", "saving", "total", "energy_used")}
            return self._observation(results), reward, self._terminated, self._truncated, info
