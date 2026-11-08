"""Run a three-interaction-round training episode, optionally scoring your own law/claims.

python -m examples.rl_episode --world 7 --clock-mode frozen
python -m examples.rl_episode --world 7 --law /abs/law.py --claims /abs/claims.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from development_worlds import catalog
from training import DiscoveryEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", choices=catalog.WORLDS, default="7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clock-mode", choices=("realtime", "frozen"), default="realtime")
    parser.add_argument("--law", type=Path)
    parser.add_argument("--claims", type=Path)
    args = parser.parse_args()
    if bool(args.law) != bool(args.claims):
        parser.error("--law and --claims must be provided together")
    env = DiscoveryEnv(clock_mode=args.clock_mode)
    observation, info = env.reset(seed=args.seed, options={"world": args.world})
    print(json.dumps({"stage": "reset", "observation": observation, "info": info}, ensure_ascii=False))
    mission = observation["results"][0]["result"]
    probe_request = {key: mission["first_light"][key]
                     for key in ("probes", "t_launch", "readout", "source_properties")}
    for interaction_round in range(1, 4):
        actions = [{"method": "archive", "params": {}}]
        if interaction_round == 1:
            actions = [{"method": "observe", "params": {"times": [0.0, .5, 1.0]}},
                       {"method": "insert_probe", "params": probe_request},
                       {"method": "archive", "params": {}}]
        elif interaction_round == 2:
            actions = [{"method": "observe", "params": {"times": [0.0, 1.0]}},
                       {"method": "archive", "params": {}}]
        elif args.law:
            actions.append({"method": "submit", "params": {"law_path": str(args.law.resolve()),
                            "claims": json.loads(args.claims.read_text(encoding="utf-8-sig"))}})
        observation, reward, terminated, truncated, info = env.step(actions)
        print(json.dumps({"interaction_round": interaction_round, "observation": observation,
                          "reward": reward, "terminated": terminated, "truncated": truncated,
                          "info": info}, ensure_ascii=False))
        if terminated or truncated:
            break



if __name__ == "__main__":
    main()
