"""Loopback-only warmed PySR worker pool with parent-enforced job deadlines.

Example: python -m shore_agent.symbolic_service serve --worker-python .venv-pysr/bin/python
Worker stdout belongs to a private log, never to the HTTP or JSON-line protocol.
"""
from __future__ import annotations

import argparse
import copy
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit
import uuid

MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 512 * 1024
MAX_AUDIT_BYTES = 256 * 1024
MAX_AXIS_AUDIT_ROWS = 160


def _scientific_metadata(value):
    """Keep bounded scientific metadata, excluding accidental worker internals."""
    if isinstance(value, dict):
        return {key: _scientific_metadata(item) for key, item in value.items()
                if isinstance(key, str) and not any(part in key.lower() for part in
                    ("path", "directory", "traceback", "hostname", "command", "host_", "log_file"))}
    if isinstance(value, list):
        return [_scientific_metadata(item) for item in value]
    return copy.deepcopy(value)


def _search_audit(raw):
    """Retain raw equation/rejection evidence before per-job files are removed.

    Normal bounded HOFs are transported whole. If a faulty worker emits excessive
    audit data, omit explicit rows with counts rather than corrupting/truncating
    returned executable formulas or losing otherwise valid candidates.
    """
    allowed = {"source", "complexity", "loss", "origin", "search_representation", "parameter_count",
               "domain_valid", "rejection_reason", "original_source"}
    result = {}
    omitted = dict(raw.get("transport_audit_omitted_rows", {}))
    if "axis_equations" in raw:
        axes = raw["axis_equations"]
        if not isinstance(axes, dict):
            raise SymbolicServiceError("invalid_worker_result", 502)
        result["axis_equations"] = {}
        for axis in ("ax", "ay"):
            rows = axes.get(axis, [])
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise SymbolicServiceError("invalid_worker_result", 502)
            result["axis_equations"][axis] = [
                {key: copy.deepcopy(value) for key, value in row.items() if key in allowed}
                for row in rows[:MAX_AXIS_AUDIT_ROWS]]
            if len(rows) > MAX_AXIS_AUDIT_ROWS:
                omitted[axis] = omitted.get(axis, 0) + len(rows) - MAX_AXIS_AUDIT_ROWS
    for key in ("rejection_counts", "rejection_examples", "candidate_selection"):
        if key in raw:
            result[key] = _scientific_metadata(raw[key])
    # Trimming is an exceptional transport guard; the normal core cap is lower.
    while len(_json_bytes(result)) > MAX_AUDIT_BYTES:
        axes = result.get("axis_equations", {})
        axis = max((a for a in ("ax", "ay") if axes.get(a)), key=lambda a: len(axes[a]), default=None)
        if axis is not None:
            axes[axis].pop()
            omitted[axis] = omitted.get(axis, 0) + 1
        elif result.get("rejection_examples"):
            result["rejection_examples"].pop()
            omitted["rejection_examples"] = omitted.get("rejection_examples", 0) + 1
        else:
            raise SymbolicServiceError("invalid_worker_result", 502)
    if omitted:
        result["transport_audit_omitted_rows"] = omitted
        result.setdefault("candidate_selection", {})["full_axis_equations_retained"] = False
    return result


class SymbolicServiceError(RuntimeError):
    def __init__(self, code, http_status=503):
        self.code = code
        self.http_status = http_status
        super().__init__(code)


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_bytes(_json_bytes(value))
    temp.replace(path)


def _read(path, maximum=MAX_RESPONSE_BYTES):
    if path.stat().st_size > maximum:
        raise SymbolicServiceError("invalid_worker_result", 502)
    data = json.loads(path.read_bytes())
    _json_bytes(data)  # reject NaN/Infinity even though json.loads accepts them
    return data


def _request(probes, budget, random_state):
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not 0 < budget <= 120:
        raise SymbolicServiceError("invalid_search_budget", 400)
    if type(random_state) is not int or not 0 <= random_state < 2**31 - 1:
        raise SymbolicServiceError("invalid_search_seed", 400)
    if not isinstance(probes, list) or not 1 <= len(probes) <= 128:
        raise SymbolicServiceError("invalid_search_probes", 400)
    ids = [p.get("probe_id") for p in probes if isinstance(p, dict)]
    if len(ids) != len(probes) or not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
        raise SymbolicServiceError("invalid_search_probes", 400)
    result = {"probes": probes, "time_budget_seconds": float(budget), "random_state": random_state}
    try:
        if len(_json_bytes(result)) > MAX_REQUEST_BYTES:
            raise ValueError("size")
    except (ValueError, TypeError, OverflowError) as exc:
        raise SymbolicServiceError("invalid_search_payload", 400) from exc
    return result, ids


def _public_result(raw, ids):
    """Allow only our structured scientific result; never forward worker paths."""
    from .expressions import Expression
    from .symbolic_regression import validate_axis_candidate
    if not isinstance(raw, dict):
        raise SymbolicServiceError("invalid_worker_result", 502)
    allowed = {"status", "backend", "target", "feature_names", "samples_per_probe",
               "dropped_constant_features", "omitted_incomplete_properties", "search_configuration",
               "warnings", "axis_search_seconds", "search_axes_completed", "rejected_candidate_pairs",
               "elapsed_seconds", "service_elapsed_seconds", "search_elapsed_seconds",
               "requested_time_budget_seconds", "worker_search_budget_seconds", "cleanup_reserve_seconds",
               "service_budget_exceeded", "worker_reap_pending", "cleanup_pending", "timing_scope",
               "search_version", "stop_reason", "data_quality", "coverage",
               "prediction_accuracy_status", "search_stages", "basis_diagnostics",
               "geometric_features", "projection_diagnostics"}
    result = {k: _scientific_metadata(v) for k, v in raw.items() if k in allowed}
    result.update(_search_audit(raw))
    result["used_probe_ids"] = list(ids)
    result["candidates"] = []
    candidates = raw.get("candidates", [])
    if not isinstance(candidates, list) or len(candidates) > 3:
        raise SymbolicServiceError("invalid_worker_result", 502)
    candidate_keys = {"law_spec", "parameter_spec", "complexity", "acceleration_search_mse_by_axis",
                      "acceleration_search_mse", "normalized_acceleration_search_loss",
                      "component_sources", "dsl_valid", "selection_role", "parameter_count",
                      "initial_domain_check", "short_rollout_check", "parameter_sharing",
                      "prediction_accuracy_status", "joint_fit_ready", "rank", "fitted_parameters",
                      "residuals", "fit_stage", "trajectory_refit_status",
                      "structural_complexity", "selection_diagnostics"}
    for candidate in candidates:
        try:
            Expression(candidate["law_spec"], candidate["parameter_spec"])
            if any(len(source) > 600 for source in candidate["law_spec"].values()):
                raise ValueError("length")
        except (KeyError, TypeError, ValueError) as exc:
            raise SymbolicServiceError("invalid_worker_result", 502) from exc
        item = {k: _scientific_metadata(v) for k, v in candidate.items() if k in candidate_keys}
        item["discovery_probe_ids"] = list(ids)
        result["candidates"].append(item)
    # Single-axis discoveries remain useful evidence when no complete pair fits
    # the existing five-parameter submission limit. Never label the zero-filled
    # other axis as a discovered force component or silently discard parameters.
    if "axis_candidates" in raw:
        axes = raw["axis_candidates"]
        if not isinstance(axes, dict) or set(axes) - {"ax", "ay"}:
            raise SymbolicServiceError("invalid_worker_result", 502)
        result["axis_candidates"] = {}
        axis_keys = {"component", "source", "law_spec", "parameter_spec", "parameter_count",
                     "loss", "complexity", "origin", "joint_fit_ready",
                     "parameter_budget_compatible", "prediction_accuracy_status", "initial_domain_check"}
        for axis in ("ax", "ay"):
            rows = axes.get(axis, [])
            if not isinstance(rows, list) or len(rows) > 3:
                raise SymbolicServiceError("invalid_worker_result", 502)
            result["axis_candidates"][axis] = []
            for row in rows:
                try:
                    validate_axis_candidate(row, axis)
                except (KeyError, TypeError, ValueError) as exc:
                    raise SymbolicServiceError("invalid_worker_result", 502) from exc
                item = {k: _scientific_metadata(v) for k, v in row.items() if k in axis_keys}
                item["discovery_probe_ids"] = list(ids)
                result["axis_candidates"][axis].append(item)
    if len(_json_bytes(result)) > MAX_RESPONSE_BYTES:
        raise SymbolicServiceError("invalid_worker_result", 502)
    return result


class _Worker:
    def __init__(self, process, directory, log):
        self.process, self.directory, self.log = process, directory, log
        self.busy = False
        self.failed = False
        self.stop_lock = threading.Lock()
        self.stopped = False
        self.reaped = threading.Event()

    def ready(self):
        return not self.failed and self.process.poll() is None and (self.directory / "ready.json").is_file()


class WorkerPool:
    def __init__(self, size=4, worker_python=None, spool_dir=None, warmup_timeout_seconds=900,
                 worker_command_factory=None, worker_env=None):
        if type(size) is not int or not 1 <= size <= 32:
            raise ValueError("pool size must be 1..32")
        self.root = Path(tempfile.mkdtemp(prefix="shore-sr-", dir=spool_dir)).resolve()
        (self.root / "jobs").mkdir(); (self.root / "workers").mkdir()
        # Keep the venv's executable path: resolving its symlink would silently
        # select the base interpreter and lose the isolated PySR environment.
        self.worker_python = str(Path(worker_python or sys.executable).absolute())
        self.warmup_timeout_seconds = warmup_timeout_seconds
        self.command_factory = worker_command_factory
        self.worker_env = dict(os.environ if worker_env is None else worker_env)
        # JuliaCall uses its own thread setting; JULIA_NUM_THREADS alone can be
        # overridden by PySR. Set both before the worker imports either package.
        self.worker_env.update(PYTHON_JULIACALL_THREADS="4", JULIA_NUM_THREADS="4",
                               PYTHON_JULIACALL_HANDLE_SIGNALS="yes", OPENBLAS_NUM_THREADS="1",
                               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        self.workers = [None] * size
        self.failed_starts = set()
        self.lock = threading.Lock()
        self.closed = False
        self.launch_threads = []
        for index in range(size):
            self._start(index)

    def _start(self, index):
        thread = threading.Thread(target=self._launch, args=(index,), daemon=True)
        with self.lock:
            if self.closed:
                return
            self.launch_threads.append(thread)
            thread.start()

    def _launch(self, index):
        directory = self.root / "workers" / uuid.uuid4().hex
        directory.mkdir()
        command = (self.command_factory(directory) if self.command_factory else
                   [self.worker_python, "-u", "-m", "shore_agent.symbolic_service", "worker",
                    "--worker-dir", str(directory)])
        log = (directory / "worker.log").open("ab", buffering=0)
        with self.lock:
            if self.closed:
                log.close()
                return
            try:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log,
                                           text=True, start_new_session=True, env=self.worker_env)
            except OSError:
                self.failed_starts.add(index)
                log.close()
                return
            worker = _Worker(process, directory, log)
            self.workers[index] = worker
            self.failed_starts.discard(index)
        deadline = time.monotonic() + self.warmup_timeout_seconds
        while not self.closed and process.poll() is None and time.monotonic() < deadline:
            if worker.ready():
                return
            time.sleep(.025)
        worker.failed = True
        self._stop(worker)

    @staticmethod
    def _stop(worker, wait_timeout=.05):
        with worker.stop_lock:
            if worker.stopped:
                return
            process = worker.process
            try:
                # Only a Popen we created with start_new_session=True is used.
                # Kill the group even if its leader exited while children live.
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=max(0, min(.05, wait_timeout)))
                worker.reaped.set()
            except subprocess.TimeoutExpired:
                def reap():
                    process.wait()
                    worker.reaped.set()
                threading.Thread(target=reap, daemon=True).start()
            if process.stdin:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            worker.log.close()
            worker.stopped = True

    def health(self):
        with self.lock:
            ready = sum(w is not None and w.ready() and not w.busy for w in self.workers)
            busy = sum(w is not None and w.busy for w in self.workers)
            failed = len(self.failed_starts) + sum(w is not None and (w.failed or w.process.poll() is not None) for w in self.workers)
            details = []
            for worker in self.workers:
                if worker is not None and worker.ready():
                    try:
                        metadata = _read(worker.directory/"ready.json", 16 * 1024)
                        details.append({key:metadata[key] for key in
                            ("actual_julia_threads", "warmup_stages", "warmup_seconds") if key in metadata})
                    except (OSError, ValueError, SymbolicServiceError):
                        details.append({})
            return {"status": "ready" if ready else "busy" if busy else "failed" if failed == len(self.workers) else "warming",
                    "pool_size": len(self.workers), "ready": ready, "busy": busy, "failed": failed,
                    "warming": max(0, len(self.workers) - ready - busy - failed),
                    "actual_julia_threads": [item.get("actual_julia_threads") for item in details],
                    "worker_warmup": details}

    def search(self, probes, time_budget_seconds, random_state=42):
        started = time.monotonic()
        payload, ids = _request(probes, time_budget_seconds, random_state)
        requested = payload["time_budget_seconds"]
        reserve = min(2.0, requested * .1)
        service_deadline = started + requested
        compute_deadline = service_deadline - reserve
        if time.monotonic() >= compute_deadline:
            elapsed = time.monotonic() - started
            return {"status": "timeout", "stop_reason": "budget_exhausted_before_dispatch",
                    "used_probe_ids": ids, "candidates": [], "search_axes_completed": [],
                    "elapsed_seconds": elapsed, "service_elapsed_seconds": elapsed,
                    "requested_time_budget_seconds": requested, "worker_search_budget_seconds": 0.0,
                    "cleanup_reserve_seconds": reserve, "service_budget_exceeded": elapsed > requested,
                    "worker_reap_pending": False, "cleanup_pending": False,
                    "timing_scope": "service_validation_dispatch_search_and_cleanup_wait; excludes HTTP and client processing"}
        with self.lock:
            if self.closed:
                raise SymbolicServiceError("service_closed")
            available = next(((i, w) for i, w in enumerate(self.workers) if w is not None and w.ready() and not w.busy), None)
            if available is None:
                raise SymbolicServiceError("symbolic_service_busy", 409)
            index, worker = available
            worker.busy = True
        directory = self.root / "jobs" / uuid.uuid4().hex
        restart = False
        result = None
        worker_budget = 0.0
        cleanup_done = threading.Event()
        try:
            directory.mkdir()
            worker_budget = max(0.0, compute_deadline - time.monotonic())
            if worker_budget <= 0:
                result = {"status": "timeout", "stop_reason": "budget_exhausted_before_dispatch",
                          "used_probe_ids": ids, "candidates": [], "search_axes_completed": []}
                return result
            payload["time_budget_seconds"] = worker_budget
            _save(directory / "request.json", payload)
            try:
                worker.process.stdin.write(json.dumps({"job_directory": str(directory)}) + "\n")
                worker.process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                restart = True
                raise SymbolicServiceError("symbolic_worker_failed", 502) from exc
            while True:
                if time.monotonic() >= compute_deadline:
                    restart = True
                    self._stop(worker, max(0, service_deadline - time.monotonic()))
                    progress = directory / "progress.json"
                    try:
                        raw = _read(progress) if progress.is_file() else {}
                        result = _public_result(raw, ids)
                    except (SymbolicServiceError, ValueError, OSError):
                        result = {"used_probe_ids": ids, "candidates": [], "search_axes_completed": []}
                    result["search_elapsed_seconds"] = result.get("elapsed_seconds")
                    result.update(status="timeout", stop_reason="hard_timeout")
                    return result
                result_path = directory / "result.json"
                if result_path.is_file():
                    raw = _read(result_path)
                    if raw.get("status") == "error":
                        raise SymbolicServiceError("symbolic_search_failed", 502)
                    result = _public_result(raw, ids)
                    result["search_elapsed_seconds"] = result.get("elapsed_seconds")
                    return result
                if worker.process.poll() is not None:
                    restart = True
                    raise SymbolicServiceError("symbolic_worker_failed", 502)
                time.sleep(min(.01, max(0, compute_deadline - time.monotonic())))
        finally:
            if restart:
                self._stop(worker, max(0, service_deadline - time.monotonic()))
            with self.lock:
                worker.busy = False
                if restart and self.workers[index] is worker:
                    self.workers[index] = None
                should_restart = restart and not self.closed
            def cleanup():
                if restart:
                    worker.reaped.wait()
                try:
                    shutil.rmtree(directory, ignore_errors=True)
                finally:
                    cleanup_done.set()
            threading.Thread(target=cleanup, daemon=True).start()
            cleanup_done.wait(timeout=min(.05, max(0, service_deadline - time.monotonic())))
            if should_restart:
                self._start(index)
            if result is not None:
                elapsed = time.monotonic() - started
                result.update(elapsed_seconds=elapsed, service_elapsed_seconds=elapsed,
                              requested_time_budget_seconds=requested, worker_search_budget_seconds=worker_budget,
                              cleanup_reserve_seconds=reserve, service_budget_exceeded=elapsed > requested,
                              worker_reap_pending=restart and not worker.reaped.is_set(),
                              cleanup_pending=not cleanup_done.is_set(),
                              timing_scope="service_validation_dispatch_search_and_cleanup_wait; excludes HTTP and client processing")
                if elapsed > requested:
                    result.update(status="timeout", stop_reason="service_budget_exceeded")

    def close(self):
        with self.lock:
            self.closed = True
            workers = list(self.workers)
            threads = list(self.launch_threads)
        for worker in workers:
            if worker is not None:
                self._stop(worker)
        for thread in threads:
            thread.join(timeout=2)


def search_remote(probes, time_budget_seconds, endpoint="http://127.0.0.1:8890", random_state=42):
    started = time.monotonic()
    result = None
    payload, ids = _request(probes, time_budget_seconds, random_state)
    try:
        parsed = urlsplit(endpoint)
        if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
                or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment
                or parsed.path not in ("", "/", "/search") or parsed.port is None):
            raise ValueError("endpoint")
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise SymbolicServiceError("invalid_symbolic_endpoint", 400) from exc
    # Do not perform DNS lookup or follow redirects, including for localhost.
    connection = http.client.HTTPConnection("::1" if parsed.hostname == "::1" else "127.0.0.1", port,
                                            timeout=float(time_budget_seconds) + 2)
    try:
        connection.request("POST", "/search", _json_bytes(payload), {"Content-Type": "application/json"})
        response = connection.getresponse()
        content = response.read(MAX_RESPONSE_BYTES + 1)
        if len(content) > MAX_RESPONSE_BYTES:
            raise SymbolicServiceError("invalid_service_response", 502)
        raw = json.loads(content)
        if response.status != 200:
            code = "symbolic_service_busy" if response.status == 409 else "symbolic_service_unavailable"
            raise SymbolicServiceError(code, response.status)
        result = _public_result(raw, ids)
        for key in ("stop_reason", "search_elapsed_seconds"):
            if key in raw:
                result[key] = raw[key]
        return result
    except SymbolicServiceError:
        raise
    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise SymbolicServiceError("symbolic_service_unavailable") from exc
    finally:
        connection.close()
        if result is not None:
            result["tool_client_wall_seconds"] = time.monotonic() - started


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _reply(self, status, value):
        body = _json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        self._reply(200, self.server.pool.health()) if self.path == "/health" else self._reply(404, {"error": "not_found"})

    def do_POST(self):
        if self.path != "/search":
            return self._reply(404, {"error": "not_found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES or self.headers.get_content_type() != "application/json":
                raise SymbolicServiceError("invalid_search_payload", 400)
            self.connection.settimeout(5)
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict) or set(body) - {"probes", "time_budget_seconds", "random_state"}:
                raise SymbolicServiceError("invalid_search_payload", 400)
            result = self.server.pool.search(body.get("probes"), body.get("time_budget_seconds"), body.get("random_state", 42))
            return self._reply(200, result)
        except SymbolicServiceError as exc:
            self._reply(exc.http_status, {"error": exc.code})
        except (OSError, ValueError, TypeError):
            self._reply(400, {"error": "invalid_search_payload"})
        except Exception:
            self._reply(500, {"error": "symbolic_service_error"})


def make_server(pool, port=8890):
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    server.daemon_threads = True
    server.pool = pool
    return server


def _command_lines(stream):
    # JuliaCall changes inherited pipe descriptors to nonblocking on import.
    # TextIOWrapper can mistake EAGAIN for EOF, so restore blocking reads before
    # every command, including after searches that touched Julia's I/O state.
    while True:
        os.set_blocking(stream.fileno(), True)
        line = stream.readline()
        if not line:
            return
        yield line


def _warmup_models(directory, model_factory=None, julia_main=None):
    started = time.monotonic()
    from .symbolic_regression import BINARY_OPERATORS, UNARY_OPERATORS, EARLY_UNARY_OPERATORS, SEARCH_MAXSIZE
    if model_factory is None:
        from pysr import PySRRegressor
        model_factory = PySRRegressor
    if julia_main is None:
        from juliacall import Main as julia_main
    import numpy as np
    actual_threads = int(julia_main.seval("Threads.nthreads()"))
    if actual_threads != 4:
        raise RuntimeError("symbolic_worker_requires_four_julia_threads")
    # Synthetic-only compilation warmup: no experiment, archive or world input.
    rng = np.random.default_rng(123)
    X = rng.uniform(.2, 1.2, size=(64, 7))
    # Warm both actual complexity budgets. Both stages use the same full
    # operator enum; their legacy labels do not imply an operator restriction.
    y = (2 * X[:, 1] ** 1.7 + np.sqrt(X[:, 0]) + np.sin(X[:, 3])
         + np.cos(X[:, 4]) + np.exp(X[:, 5]) + np.log(X[:, 6]))
    stages = []
    for name, unary, maxsize in (("algebra", list(EARLY_UNARY_OPERATORS), 24),
                                 ("full", list(UNARY_OPERATORS), SEARCH_MAXSIZE)):
        begin = time.monotonic()
        model = model_factory(binary_operators=list(BINARY_OPERATORS), unary_operators=unary,
                              constraints={"^": (-1, 1)}, maxsize=maxsize,
                              niterations=1, populations=2, population_size=27,
                              ncycles_per_iteration=10, timeout_in_seconds=2, parallelism="multithreading",
                              warm_start=False, precision=64, verbosity=0, progress=False,
                              output_directory=str(directory / "warmup" / name))
        model.fit(X, y, weights=np.ones(len(y)), variable_names=["t", "x", "y", "vx", "vy", "mass", "charge"])
        stages.append({"stage":name,"maxsize":maxsize,"unary_operators":unary,
                       "elapsed_seconds":time.monotonic()-begin})
        del model
    return {"actual_julia_threads":actual_threads,"warmup_stages":stages,
            "warmup_seconds":time.monotonic()-started}


def _worker(directory):
    from .symbolic_regression import run_search
    directory = Path(directory).resolve()
    warmup = _warmup_models(directory)
    _save(directory / "ready.json", dict(warmup, ready=True))
    jobs_root = directory.parent.parent / "jobs"
    for line in _command_lines(sys.stdin):
        job = None
        try:
            command = json.loads(line)
            job = Path(command["job_directory"]).resolve()
            if job.parent != jobs_root or not job.is_dir():
                raise ValueError("invalid job directory")
            payload = _read(job / "request.json", MAX_REQUEST_BYTES)
            result = run_search(payload["probes"], payload["time_budget_seconds"],
                                random_state=payload["random_state"], output_directory=job / "outputs",
                                progress_callback=lambda report: _save(job / "progress.json", report))
            _save(job / "result.json", result)
        except Exception:
            if job is not None and job.parent == jobs_root and job.is_dir():
                _save(job / "result.json", {"status": "error", "error_code": "search_failed"})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8890)
    serve.add_argument("--pool-size", type=int, default=4)
    serve.add_argument("--worker-python", default=sys.executable)
    serve.add_argument("--spool-dir", type=Path)
    worker = sub.add_parser("worker")
    worker.add_argument("--worker-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        return _worker(args.worker_dir)
    pool = WorkerPool(args.pool_size, args.worker_python, args.spool_dir)
    server = make_server(pool, args.port)
    def stop(_signum, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=.1)
    finally:
        server.server_close()
        pool.close()


if __name__ == "__main__":
    main()
