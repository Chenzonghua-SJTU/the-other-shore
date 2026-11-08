"""Stream reproducible, noisy exploration episodes (no oracle labels) to JSONL.

Usage: python -m training.generate --episodes 20 --seed 42 --output data/episodes.jsonl
The companion .manifest.json contains trainer-only world IDs and noise seeds.
This random policy gathers observations; it neither fits laws nor trains an RL
model, and its three-interaction-round episodes terminate with zero reward without submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from development_worlds import catalog, evaluation
from training import DiscoveryEnv


def random_launch(mission, rng):
    """Build a legal experiment using only mission controls and input bounds."""
    region = mission["launch_region"]
    controls = mission["controls"]
    probes = []
    for _ in range(2):
        angle = float(rng.uniform(0, 2 * np.pi))
        radius = float(rng.uniform(region["radius_min"], region["radius_max"]))
        unit = np.array([np.cos(angle), np.sin(angle)])
        # Mild tangential launches keep the original nonsmooth radial worlds
        # away from deliberate origin-crossing stress tests in this basic demo.
        speed = float(rng.uniform(.15, .3) * mission["initial_speed_max"])
        probes.append({"pos": (np.asarray(region["center"]) + radius * unit).tolist(),
                       "vel": (speed * np.array([-unit[1], unit[0]])).tolist(),
                       "properties": {key: float(rng.uniform(rule["min"], rule["max"]))
                                      for key, rule in controls["probe_properties"].items()}})
    duration_ticks = int(round(min(2.0, mission["maximum_single_trajectory_duration"]) / .005))
    latest_tick = int(round(mission["maximum_physical_time"] / .005)) - duration_ticks
    launch_tick = int(rng.integers(0, latest_tick + 1))
    readout = (np.unique(np.linspace(0, duration_ticks, 12, dtype=int)) * .005).tolist()
    return {"method": "insert_probe", "params": {"probes": probes,
            "t_launch": launch_tick * .005, "readout": readout,
            "source_properties": {key: float(rng.uniform(rule["min"], rule["max"]))
                                  for key, rule in controls["source_properties"].items()}}}


def generate(output, *, episodes=20, seed=42, worlds=None):
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    output = Path(output)
    manifest_path = Path(str(output) + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Output or companion manifest already exists; choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = DiscoveryEnv(worlds, clock_mode="frozen", anonymize_world_id=True)
    rng = np.random.default_rng(seed)
    manifest = {"format": "shore-exploration-jsonl-1.0", "status": "running",
                "seed": seed, "clock_mode": "frozen", "episodes_requested": episodes,
                "purpose": "noisy exploration only; no fitted laws or nonzero rewards",
                "worlds": list(env.worlds), "world_sampling": "seeded permutation per coverage block",
                "episodes": [],
                "source_sha256": evaluation._source_hashes()}
    for name in ("env.py", "generate.py"):
        manifest["source_sha256"]["training/" + name] = hashlib.sha256(
            (Path(__file__).parent / name).read_bytes()).hexdigest()
    # Exclusive creation protects both previous data and metadata.
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    try:
        with output.open("x", encoding="utf-8") as stream:
            for index in range(episodes):
                # Shuffle each coverage block so the public episode_index does
                # not encode the supposedly anonymized world ID by modulo.
                if index % len(env.worlds) == 0:
                    world_order = rng.permutation(env.worlds).tolist()
                world = world_order[index % len(env.worlds)]
                episode_seed = int(rng.integers(0, 2**63 - 1))
                initial, _ = env.reset(seed=episode_seed, options={"world": world})
                mission = initial["results"][0]["result"]
                # Each item is one environment interaction round. Multiple
                # tools in its list execute in sequence within that one round.
                actions = [[{"method": "observe", "params": {"times": [0.0, .5, 1.0]}},
                            random_launch(mission, rng), {"method": "archive", "params": {}}],
                           [{"method": "observe", "params": {"times": [0.0, 1.0]}},
                            {"method": "archive", "params": {}}],
                           [{"method": "archive", "params": {}}]]
                record = {"episode_index": index, "initial_observation": initial, "transitions": []}
                for action in actions:
                    observation, reward, terminated, truncated, info = env.step(action)
                    if not all(item["ok"] for item in observation["results"]):
                        raise RuntimeError(f"Generated an invalid experiment in episode {index}")
                    record["transitions"].append({"action": action, "observation": observation,
                                                  "reward": reward, "terminated": terminated,
                                                  "truncated": truncated, "info": info})
                    if terminated or truncated:
                        break
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                manifest["episodes"].append({"episode_index": index, "reset_seed": episode_seed,
                                             **env.trainer_metadata})
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "incomplete"
        raise
    finally:
        manifest["episodes_written"] = len(manifest["episodes"])
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2)
                                 + "\n", encoding="utf-8")
    return {"output": str(output.resolve()), "manifest": str(manifest_path.resolve()),
            "episodes": episodes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worlds", nargs="+", choices=catalog.WORLDS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.output, episodes=args.episodes, seed=args.seed, worlds=args.worlds),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
