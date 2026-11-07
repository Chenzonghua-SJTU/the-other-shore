"""E10 harness: original official runner protocol, experimental noise constructor only.
Scientific Agent and all official files remain unchanged. Clean arm is an ablation.
"""
import json,os,subprocess,time
from pathlib import Path
from development_worlds.runner import ROOT,DevelopmentConsole,DevelopmentSession,serve_session,catalog,evaluate_law,_stop

def make_console(world,seed,noise):
    sigma=catalog.noise_sigma(world) if noise else 0.
    console=DevelopmentConsole(world,sigma,seed,catalog.config(world))
    if not noise:console._velocity_sigma=0.
    original=console.mission
    def mission():
        result=original()
        result['noise']['position_sigma']=sigma
        return result
    console.mission=mission
    return console

def run_episode(world, directory, command, *, seed=0, startup_seconds=900.,
                evaluate=True, cwd=None, noise=True):
    """Start one agent process; judge energy from the session, never self-report.

    ``command`` is a list of executable arguments, never a shell string. The
    caller can supply a different editable agent with the same Console API.
    A new directory is required to preserve earlier experiments and results.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    console = make_console(world, seed, noise)
    (directory/"actual-noise.json").write_text(json.dumps({"noise":noise,"position_sigma":console._sigma,"velocity_sigma":console._velocity_sigma}))
    session = DevelopmentSession(console, directory/"session")
    environment = os.environ.copy()
    environment["RIFT_OUTPUT_DIR"] = str(directory/"agent-output")
    environment["PYTHONPATH"] = str(ROOT)+os.pathsep+environment.get("PYTHONPATH", "")
    launched = time.monotonic()
    reason, exit_code = "agent_exited", None
    with serve_session(session) as credentials, (directory/"agent.log").open("w", encoding="utf-8") as log:
        environment.update(RIFT_CONSOLE_URL=credentials["endpoint"], RIFT_SESSION_TOKEN=credentials["token"])
        process = subprocess.Popen(list(command), cwd=str(cwd or ROOT), env=environment,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
        try:
            while process.poll() is None:
                snapshot = session.snapshot()
                if snapshot["closed"]:
                    reason = "submitted"
                    break
                if snapshot["started"] and snapshot["budget"]["wall_clock_remaining_seconds"] <= 0:
                    reason = "timeout"
                    break
                if not snapshot["started"] and time.monotonic()-launched >= startup_seconds:
                    reason = "startup_timeout"
                    break
                time.sleep(.05)
            exit_code = process.poll()
        finally:
            _stop(process, graceful=reason == "submitted")
            exit_code = process.returncode
        finalized = session.finalize(reason)
    result = {"world": str(world), "scope": "public_development", "end_reason": finalized["reason"],
              "agent_exit_code": exit_code, "budget": finalized["budget"],
              "has_valid_submission": finalized["has_valid_submission"],
              "wall_clock_seconds": catalog.WALL_CLOCK_SECONDS, "startup_and_exploration_elapsed_seconds": time.monotonic()-launched}
    if session.latest is not None:
        result["law_sha256"] = session.latest["law_sha256"]
        result["claims_sha256"] = session.latest["claims_sha256"]
        if evaluate:
            pair = Path(session.latest["artifact_directory"])
            claims = json.loads((pair/"claims.json").read_text(encoding="utf-8-sig"))
            evaluation_start = time.monotonic()
            result["score"] = evaluate_law(world, pair/"law.py", claims, finalized["budget"]["energy_used"])
            result["evaluation_elapsed_seconds"] = time.monotonic()-evaluation_start
    elif evaluate:
        result["score"] = {"total": 0., "reason": "no_valid_submission"}
    (directory/"result.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding="utf-8")
    return result
