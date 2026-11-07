"""Small interpreted acceleration DSL. No eval, exec, attributes or indexing.

Physical coefficients/offsets belong in parameter_spec. Literal 0 and 1 are
structural constants; integers -3..3 may be literal exponents. Constant arithmetic
and adding/subtracting nonzero literals are forbidden (write a named offset).
This is a bounded mathematical language, not a general Python sandbox.
"""
from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping

import numpy as np


class ExpressionError(ValueError):
    code = "invalid_expression"


class BodyTimeCoverageError(ExpressionError):
    code = "body_time_coverage"


STATE_NAMES = {"t", "x", "y", "vx", "vy"}
BODY_NAMES = {"bx", "by"}
FUNCTIONS = {"sin": math.sin, "cos": math.cos, "exp": math.exp,
             "sqrt": math.sqrt, "abs": abs, "log": math.log}
RESERVED = STATE_NAMES | BODY_NAMES | set(FUNCTIONS) | {"q", "sum_b"}
ALLOWED_PROPERTIES = frozenset({"mass", "charge"})
MAX_PARAMETERS = 5
MAX_NODES = 256
MAX_DEPTH = 16
MAX_MAGNITUDE = 1e12


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExpressionError(f"{label} must be a finite number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ExpressionError(f"{label} must be a finite number") from exc
    if not math.isfinite(value) or abs(value) > MAX_MAGNITUDE:
        raise ExpressionError(f"{label} must be finite and within +/- {MAX_MAGNITUDE:g}")
    return value


def validate_parameter_spec(parameter_spec):
    if not isinstance(parameter_spec, list) or len(parameter_spec) > MAX_PARAMETERS:
        raise ExpressionError("parameter_spec must be a list with at most 5 scalar parameters")
    names = set()
    normalized = []
    for item in parameter_spec:
        if not isinstance(item, dict) or set(item) - {"name", "initial", "lower", "upper"}:
            raise ExpressionError("parameter entries allow name, initial, lower, upper only")
        name = item.get("name")
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", name)
                or name in RESERVED or name in names or "__" in name):
            raise ExpressionError("invalid, reserved or duplicate parameter name")
        if "initial" not in item:
            raise ExpressionError(f"missing initial value for {name}")
        initial = _number(item["initial"], f"{name}.initial")
        lower = _number(item.get("lower", -MAX_MAGNITUDE), f"{name}.lower")
        upper = _number(item.get("upper", MAX_MAGNITUDE), f"{name}.upper")
        if not lower < upper or not lower <= initial <= upper:
            raise ExpressionError(f"{name}: require lower < upper and lower <= initial <= upper")
        normalized.append({"name": name, "initial": initial, "lower": lower, "upper": upper})
        names.add(name)
    return normalized


def body_positions(bodies, t=None):
    """Resolve supplied visible positions at absolute time; never query a simulator.

    Historical lists represent static positions. An observed trajectory context
    is interpolated only inside its acquired time range, without extrapolation.
    """
    if isinstance(bodies, Mapping) and bodies.get('kind') == 'observed_body_trajectories':
        try:
            times = np.asarray(bodies['times'], dtype=float)
            ids = bodies['body_ids']
            observed = np.asarray(bodies['positions'], dtype=float)
            if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
                    or np.any(np.diff(times) <= 0) or not isinstance(ids, (list, tuple))
                    or not 1 <= len(ids) <= 64 or not all(isinstance(i, str) for i in ids)
                    or len(set(ids)) != len(ids)
                    or observed.shape != (len(times), len(ids), 2)
                    or not np.isfinite(observed).all()):
                raise ExpressionError('Invalid observed body trajectory context')
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ExpressionError):
                raise
            raise ExpressionError('Invalid observed body trajectory context') from exc
        if t is None:
            raise BodyTimeCoverageError('body_time_coverage: absolute time is required for observed body trajectories')
        t = _number(t, 'body time')
        tolerance = 1e-10
        if t < times[0] - tolerance or t > times[-1] + tolerance:
            raise BodyTimeCoverageError(
                f'body_time_coverage: t={t:g} is outside acquired body times '
                f'[{times[0]:g}, {times[-1]:g}]; obtain observe_body data covering this interval')
        t = min(max(t, float(times[0])), float(times[-1]))
        right = int(np.searchsorted(times, t, side='right'))
        if right == 0 or len(times) == 1:
            return observed[0].copy()
        if right == len(times):
            return observed[-1].copy()
        left = right - 1
        weight = (t - times[left]) / (times[right] - times[left])
        return observed[left] * (1 - weight) + observed[right] * weight
    if isinstance(bodies, Mapping):
        bodies = list(bodies.values())
    if not isinstance(bodies, (list, tuple, np.ndarray)) or len(bodies) > 64:
        raise ExpressionError("bodies must contain at most 64 supplied visible positions")
    positions = []
    for body in bodies:
        value = body.get("position") if isinstance(body, Mapping) else body
        try:
            position = np.asarray(value, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ExpressionError("invalid body position") from exc
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ExpressionError("body positions must be finite vectors of length 2")
        positions.append(position)
    return np.asarray(positions, dtype=float).reshape((-1, 2))


def _literal(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal(node.operand)
        if value is not None:
            return -value if isinstance(node.op, ast.USub) else value
    return None


class Expression:
    def __init__(self, law_spec, parameter_spec, allowed_properties=ALLOWED_PROPERTIES):
        if not isinstance(law_spec, dict) or set(law_spec) != {"ax", "ay"}:
            raise ExpressionError("law_spec must contain exactly ax and ay strings")
        self.law_spec = dict(law_spec)
        self.parameter_spec = validate_parameter_spec(parameter_spec)
        self.parameter_names = {p["name"] for p in self.parameter_spec}
        self.allowed_properties = frozenset(allowed_properties)
        self.required_properties = set()
        self.uses_bodies = False
        self.used_parameters = set()
        self.nodes = []
        total = 0
        for component in ("ax", "ay"):
            source = law_spec[component]
            if not isinstance(source, str) or not source or len(source) > 1024:
                raise ExpressionError("each formula must be a nonempty string of at most 1024 characters")
            try:
                tree = ast.parse(source, mode="eval")
            except (SyntaxError, RecursionError) as exc:
                raise ExpressionError("formula is not valid DSL syntax") from exc
            total += sum(1 for _ in ast.walk(tree))
            if total > MAX_NODES:
                raise ExpressionError(
                    f"formula exceeds {MAX_NODES} total AST nodes ({total} counted through {component}); "
                    "factor repeated terms or combine body sums. This is a syntax/resource limit, "
                    "not evidence against the physical hypothesis.")
            self._check(tree.body, 0, False)
            self.nodes.append(tree.body)
        unused = self.parameter_names - self.used_parameters
        if unused:
            raise ExpressionError("unused declared parameters: " + ", ".join(sorted(unused)))

    def _check(self, node, depth, in_body, exponent=False):
        if depth > MAX_DEPTH:
            raise ExpressionError("formula nesting is too deep")
        literal = _literal(node)
        if literal is not None:
            if type(literal) is not int or (literal not in range(-3, 4) if exponent else literal not in (0, 1, -1)):
                raise ExpressionError("use named parameters for constants; only 0, +/-1 and integer exponents -3..3 are structural literals")
            return False
        if isinstance(node, ast.Name):
            if node.id in self.parameter_names:
                self.used_parameters.add(node.id)
            elif node.id not in STATE_NAMES and not (in_body and node.id in BODY_NAMES):
                raise ExpressionError(f"unknown or out-of-scope symbol {node.id!r}")
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return self._check(node.operand, depth + 1, in_body)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            left = self._check(node.left, depth + 1, in_body)
            right = self._check(node.right, depth + 1, in_body, isinstance(node.op, ast.Pow))
            if not left and not right:
                raise ExpressionError("constant arithmetic is forbidden; declare a parameter")
            if isinstance(node.op, (ast.Add, ast.Sub)):
                if any(_literal(n) not in (None, 0) for n in (node.left, node.right)):
                    raise ExpressionError("numeric offsets must be named parameters")
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if len(node.args) != 1:
                raise ExpressionError("functions take exactly one positional argument")
            if name == "q":
                key = node.args[0]
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str) or key.value not in self.allowed_properties:
                    raise ExpressionError("q accepts one allowed property name string")
                self.required_properties.add(key.value)
                return True
            if name == "sum_b":
                if in_body:
                    raise ExpressionError("nested body sums are forbidden")
                self.uses_bodies = True
                if not self._check(node.args[0], depth + 1, True):
                    raise ExpressionError("constant body sums are forbidden")
                return True
            if name in FUNCTIONS:
                if not self._check(node.args[0], depth + 1, in_body):
                    raise ExpressionError("constant function expressions are forbidden")
                return True
        raise ExpressionError(f"unsupported DSL syntax: {type(node).__name__}")

    def _evaluate(self, node, context, positions, properties):
        if isinstance(node, ast.Constant):
            value = float(node.value)
        elif isinstance(node, ast.Name):
            value = context[node.id]
        elif isinstance(node, ast.UnaryOp):
            value = self._evaluate(node.operand, context, positions, properties)
            if isinstance(node.op, ast.USub):
                value = -value
        elif isinstance(node, ast.BinOp):
            a = self._evaluate(node.left, context, positions, properties)
            b = self._evaluate(node.right, context, positions, properties)
            if isinstance(node.op, ast.Add): value = a + b
            elif isinstance(node.op, ast.Sub): value = a - b
            elif isinstance(node.op, ast.Mult): value = a * b
            elif isinstance(node.op, ast.Div): value = a / b
            else:
                if abs(b) > 32:
                    raise ExpressionError("exponent magnitude exceeds 32")
                value = math.pow(a, b)
        else:
            name = node.func.id
            if name == "q":
                value = properties[node.args[0].value]
            elif name == "sum_b":
                value = 0.0
                for bx, by in positions:
                    value += self._evaluate(node.args[0], {**context, "bx": float(bx), "by": float(by)}, positions, properties)
            else:
                value = FUNCTIONS[name](self._evaluate(node.args[0], context, positions, properties))
        if not math.isfinite(value) or abs(value) > MAX_MAGNITUDE:
            raise ExpressionError("non-finite or excessive intermediate expression value")
        return value

    def acceleration(self, t, r, v, bodies, properties, parameters):
        if not isinstance(parameters, Mapping) or set(parameters) != self.parameter_names:
            raise ExpressionError("parameter values must match declared names exactly")
        context = {name: _number(value, name) for name, value in parameters.items()}
        try:
            r, v = np.asarray(r, dtype=float), np.asarray(v, dtype=float)
            if r.shape != (2,) or v.shape != (2,):
                raise ExpressionError("r and v must have two components")
            for name, value in zip(("t", "x", "y", "vx", "vy"), (t, *r, *v)):
                context[name] = _number(float(value), name)
            if not isinstance(properties, Mapping):
                raise ExpressionError("properties must be a mapping")
            if self.required_properties - properties.keys():
                raise ExpressionError("formula requires a property absent from this probe")
            props = {name: _number(properties[name], name) for name in self.required_properties}
            positions = body_positions(bodies, t) if self.uses_bodies else np.empty((0, 2))
            return np.asarray([self._evaluate(node, context, positions, props) for node in self.nodes])
        except ExpressionError:
            raise
        except (ValueError, OverflowError, ZeroDivisionError, TypeError, KeyError) as exc:
            raise ExpressionError(f"expression domain/evaluation error: {type(exc).__name__}") from exc
