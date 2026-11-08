"""Twenty public force fields for a synthetic training extension of starter v15.

All fields act on independent passive probes in two dimensions.  Each source
or origin marker is fixed; visible tracers exert no force. Worlds 15, 16 and
20 contain fixed invisible sources; world 14 has two fixed visible sources.
The fixed public integrator is shared with candidate prediction.  This module
supplies clean trajectories; the separate console supplies noise and budgets.
No external physics package, private simulator or private configuration is read.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from rift_numerics import DT, NUMERICS_VERSION, diagnostic_rollout, rollout, sample_indices

from .world_specs import WORLDS, SCALAR_WORLDS, WORLD_SPECS

VERSION = "shore-worlds-20-1.0.0"
EPSILON = 0.05
SINGLE_CENTER = SCALAR_WORLDS
PARAMETERS = {
    "1": {"G": 1.0},
    "2": {"k": 0.06},
    "3": {"kx": 0.04, "ky": 0.12, "angle": float(np.pi / 6)},
    "4": {"G": 2.0, "lambda": 4.0},
    "5": {"a": 0.06, "b": 0.10, "c": 0.002, "d": 0.012},
    "6": {"kx": 0.06, "ky": 0.18, "period": 16.0, "phase": 0.3},
    "7": {"G": 1.2, "s": 0.8},
    "8": {"k": 0.14, "L": 2.0},
    "9": {"k": 0.03, "c": 0.004},
    "10": {"a": 0.08, "b": 0.008, "ky": 0.12},
    "11": {"k": 0.025, "A": 0.10, "q": 1.3},
    "12": {"A": 0.18, "s": 2.0},
    "13": {"k": 0.008, "radius": 2.5},
    "14": {"G1": 0.9, "G2": 1.4, "s": 0.8, "k": 0.025},
    "15": {"G": 1.2, "s": 0.8, "cx": 1.1, "cy": -0.9, "k": 0.025},
    "16": {"G1": 0.8, "G2": 1.1, "s": 0.9, "d": 1.7, "k": 0.025},
    "17": {"k": 0.10, "amplitude": 0.35, "period": 12.0, "phase": 0.4},
    "18": {"k": 0.08, "A": 0.12, "period": 10.0},
    "19": {"k": 0.06, "A": 0.10, "q": 1.2, "period": 14.0, "phase": 0.4},
    "20": {"k": 0.08, "amplitude": 0.3, "period": 10.0, "cx": 1.2, "cy": -0.7},
}


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_, str, bytes)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite scalar.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _tick(value, name, *, dt=DT):
    value = _finite_scalar(value, name)
    step = _finite_scalar(dt, "dt")
    if value < 0 or step <= 0:
        raise ValueError(f"{name} must be nonnegative and dt must be positive.")
    ratio = value / step
    if not np.isfinite(ratio) or ratio > np.iinfo(np.int64).max:
        raise ValueError(f"{name} is outside the supported time range.")
    tick = round(ratio)
    if abs(tick - ratio) > 1e-9:
        raise ValueError(f"{name} must lie on the {step} time grid.")
    return tick


def _sample_ticks(times, name, *, limit=50):
    if isinstance(times, (str, bytes)):
        raise ValueError(f"{name} must be a sequence.")
    try:
        values = list(times)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence.") from exc
    if not 1 <= len(values) <= limit:
        raise ValueError(f"{name} requires 1 to {limit} entries.")
    ticks = np.array([_tick(t, name) for t in values], dtype=np.int64)
    if np.any(np.diff(ticks) <= 0):
        raise ValueError(f"{name} must be strictly increasing.")
    return ticks


def _state_array(value, name):
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite array of shape (N, 2).") from exc
    if result.ndim != 2 or result.shape[1] != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite array of shape (N, 2).")
    return result


class DevelopmentSimulator:
    """One public, fixed-parameter development force field.

    ``integrate(..., dt=...)`` is a clean numerical diagnostic for convergence
    checks.  The experiment methods ``launch`` and ``background`` always use
    the frozen 0.005 step.  Input-domain and per-episode quota enforcement are
    supplied by the console.  Direct clean simulator access is deliberately
    available for these public development worlds.
    """

    def __init__(self, world):
        if isinstance(world, (bool, np.bool_)) or str(world) not in WORLDS:
            raise ValueError("Development world must be 1 through 20.")
        self.world = str(world)
        self.parameters = copy.deepcopy(PARAMETERS[self.world])
        self.parameter_count = len(self.parameters)
        self.claims_truth = copy.deepcopy(WORLD_SPECS[self.world]["claims_truth"])
        self.cache_token = hashlib.sha256(
            Path(__file__).read_bytes() + Path(__file__).with_name("world_specs.py").read_bytes() + json.dumps(
                {"world": self.world, "parameters": self.parameters,
                 "numerics_version": NUMERICS_VERSION, "dt": DT},
                sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _properties(self, n, source_properties, probe_properties):
        source = {} if source_properties is None else source_properties
        expected = {"p1"} if self.world in SINGLE_CENTER else set()
        if not isinstance(source, dict) or set(source) != expected:
            raise ValueError(f"source_properties requires exactly {sorted(expected)}.")
        p1 = _finite_scalar(source["p1"], "p1") if expected else None
        if p1 is not None and p1 <= 0:
            raise ValueError("p1 must be positive.")
        if isinstance(probe_properties, dict):
            properties = [probe_properties] * n
        elif isinstance(probe_properties, (list, tuple)) and len(probe_properties) == n:
            properties = probe_properties
        else:
            raise ValueError("probe_properties requires an object or one object per probe.")
        key = "p2" if self.world in SINGLE_CENTER else "mass"
        values = np.empty(n, dtype=np.float64)
        for i, item in enumerate(properties):
            if not isinstance(item, dict) or set(item) != {key}:
                raise ValueError(f"Probe properties require exactly {key}.")
            values[i] = _finite_scalar(item[key], key)
        if np.any(values <= 0):
            raise ValueError(f"{key} must be positive.")
        return p1, values

    @staticmethod
    def _rotated_linear(positions, kx, ky, angle):
        """Apply -R(angle) diag(kx, ky) R(angle)^T in global coordinates."""
        c, s = np.cos(angle), np.sin(angle)
        xx = kx * c*c + ky * s*s
        yy = kx * s*s + ky * c*c
        xy = (kx - ky) * c*s
        x, y = positions[:, 0], positions[:, 1]
        return -np.column_stack((xx*x + xy*y, xy*x + yy*y))

    def _test_acceleration(self, t, positions, p1, values):
        p = self.parameters
        if self.world in ("1", "4"):
            radius = np.linalg.norm(positions, axis=1)
            rho = np.sqrt(radius*radius + EPSILON*EPSILON)
            unit = np.divide(positions, radius[:, None], out=np.zeros_like(positions),
                             where=radius[:, None] > 0)
            if self.world == "1":
                amplitude = p["G"] * p1 / (4*np.pi*values*rho)
            else:
                amplitude = p["G"] * p1 * np.exp(-rho/p["lambda"]) / (2*np.pi*values*rho)
            return -amplitude[:, None] * unit
        if self.world == "2":
            return -(p["k"] * p1/values)[:, None] * positions
        if self.world == "3":
            return self._rotated_linear(positions, p["kx"], p["ky"], p["angle"])
        if self.world == "5":
            x, y = positions[:, 0], positions[:, 1]
            return -np.column_stack((p["a"]*x + p["c"]*x**3 + p["d"]*x*y*y,
                                     p["b"]*y + p["c"]*y**3 + p["d"]*y*x*x))
        if self.world == "6":
            # A half-turn restores the anisotropy matrix: field period=period.
            angle = np.pi*t/p["period"] + p["phase"]
            return self._rotated_linear(positions, p["kx"], p["ky"], angle)
        radius2 = np.sum(positions*positions, axis=1)
        if self.world == "7":
            return -(p["G"]*p1/values/(radius2+p["s"]**2)**1.5)[:, None]*positions
        if self.world == "8":
            return -(p["k"]*p1/values/np.sqrt(1+radius2/p["L"]**2))[:, None]*positions
        if self.world == "9":
            return -(p1/values*(p["k"]+p["c"]*radius2))[:, None]*positions
        if self.world == "10":
            x, y = positions[:, 0], positions[:, 1]
            return np.column_stack((p["a"]*x-p["b"]*x**3, -p["ky"]*y))
        if self.world == "11":
            return -p["k"]*positions-p["A"]*p["q"]*np.sin(p["q"]*positions)
        if self.world == "12":
            return -(p1/values*p["A"]*np.exp(-radius2/(2*p["s"]**2)))[:, None]*positions
        if self.world == "13":
            return -(p["k"]*(radius2-p["radius"]**2))[:, None]*positions
        if self.world in ("14", "15", "16"):
            result = -p["k"]*positions
            if self.world == "15":
                sources = ((p["G"], np.array([p["cx"], p["cy"]])),)
            else:
                d = 2.0 if self.world == "14" else p["d"]
                sources = ((p["G1"], np.array([-d, 0.])),
                           (p["G2"], np.array([d, 0.])))
            for strength, center in sources:
                displacement = positions-center
                distance2 = np.sum(displacement*displacement, axis=1)
                result = result-strength*displacement/(distance2+p["s"]**2)[:, None]**1.5
            return result
        if self.world == "17":
            return -p["k"]*(1+p["amplitude"]*np.sin(2*np.pi*t/p["period"]+p["phase"]))*positions
        if self.world == "18":
            angle = 2*np.pi*t/p["period"]
            return -p["k"]*positions+p["A"]*np.array([np.cos(angle), np.sin(angle)])
        if self.world == "19":
            result = -p["k"]*positions
            result[:, 0] += p["A"]*p["q"]*np.sin(p["q"]*positions[:, 0]-2*np.pi*t/p["period"]+p["phase"])
            return result
        if self.world == "20":
            center = np.array([p["cx"], p["cy"]])
            return -p["k"]*(1+p["amplitude"]*np.sin(2*np.pi*t/p["period"]))*(positions-center)
        raise AssertionError("Unregistered force field.")

    def acceleration(self, t, positions, *, source_properties=None, probe_properties=None):
        instant = _finite_scalar(t, "t")
        positions = _state_array(positions, "positions")
        p1, values = self._properties(len(positions), source_properties, probe_properties)
        result = self._test_acceleration(instant, positions, p1, values)
        if not np.isfinite(result).all():
            raise FloatingPointError("Acceleration produced a nonfinite state.")
        return result

    def integrate(self, positions, velocities, *, t_launch=0., duration=1.,
                  source_properties=None, probe_properties=None, dt=DT):
        """Clean full history; a nondefault dt is for convergence diagnostics."""
        _tick(t_launch, "t_launch")
        positions = _state_array(positions, "positions")
        velocities = _state_array(velocities, "velocities")
        if positions.shape != velocities.shape:
            raise ValueError("Positions and velocities must have identical shape.")
        p1, values = self._properties(len(positions), source_properties, probe_properties)
        callback = lambda t, x: self._test_acceleration(t, x, p1, values)
        if dt == DT:
            return rollout(callback, positions, velocities, t_launch=t_launch, duration=duration)
        return diagnostic_rollout(callback, positions, velocities, t_launch=t_launch,
                                  duration=duration, dt=dt)

    def background_initial_state(self):
        if self.world == "14":
            return {"positions": np.array([[-2., 0.], [2., 0.]]),
                    "velocities": np.zeros((2, 2)), "ids": ["N0", "N1"]}
        if self.world in SINGLE_CENTER:
            return {"positions": np.zeros((1, 2)), "velocities": np.zeros((1, 2)), "ids": ["N0"]}
        k = np.arange(8, dtype=np.float64)
        angle = 2*np.pi*k/8
        radial = np.column_stack((np.cos(angle), np.sin(angle)))
        tangent = np.column_stack((-np.sin(angle), np.cos(angle)))
        return {"positions": np.vstack((np.zeros((1, 2)), (2+.15*k)[:, None]*radial)),
                "velocities": np.vstack((np.zeros((1, 2)), (.4+.03*k)[:, None]*tangent)),
                "ids": [f"N{i}" for i in range(9)]}

    def _background_acceleration(self, t, positions):
        if self.world in SINGLE_CENTER or self.world == "14":
            return np.zeros_like(positions)
        result = self._test_acceleration(t, positions, None, np.ones(len(positions)))
        result[0] = 0.
        return result

    def background(self, times):
        """Clean visible background, replayed from the global physical epoch."""
        ticks = _sample_ticks(times, "times")
        actual = ticks*DT
        state = self.background_initial_state()
        history = rollout(self._background_acceleration, state["positions"], state["velocities"],
                          t_launch=0., duration=float(actual[-1]))
        return {"times": actual, "positions": history["positions"][ticks],
                "velocities": history["velocities"][ticks], "ids": state["ids"]}

    def launch(self, probes, t_launch, readout, source_properties=None):
        """Clean fixed-step experiment; returns (readouts, probes, 2) arrays.

        Background is a separate observation operation.  Launch count, energy,
        allowed initial-state ranges and observation noise belong to the console.
        """
        if not isinstance(probes, (list, tuple)) or not 1 <= len(probes) <= 8:
            raise ValueError("A launch requires 1 to 8 probes.")
        if any(not isinstance(p, dict) or set(p) != {"pos", "vel", "properties"} for p in probes):
            raise ValueError("Each probe requires exactly pos, vel and properties.")
        launch_tick = _tick(t_launch, "t_launch")
        offsets = _sample_ticks(readout, "readout")
        actual_launch, actual_readout = launch_tick*DT, offsets*DT
        history = self.integrate([p["pos"] for p in probes], [p["vel"] for p in probes],
            t_launch=actual_launch, duration=float(actual_readout[-1]),
            source_properties=source_properties, probe_properties=[p["properties"] for p in probes])
        return {"times": (launch_tick+offsets)*DT, "t_launch": actual_launch,
                "readout": actual_readout, "positions": history["positions"][offsets],
                "velocities": history["velocities"][offsets]}


__all__ = ["DevelopmentSimulator", "DT", "NUMERICS_VERSION", "EPSILON", "SINGLE_CENTER",
           "WORLDS", "PARAMETERS"]
