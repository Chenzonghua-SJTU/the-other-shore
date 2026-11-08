"""Validate model-authored background dynamics without selecting a hypothesis.

Background expressions share the probe candidate's declared scalar parameters.
No geometry, motion class, coefficients or scientific defaults are inferred here.
"""
import ast
import copy

from ..repository.shore_agent.expressions import Expression, validate_parameter_spec


class BackgroundModelError(ValueError):
    code = 'background_model_required'


def explicit_background(expression, background_law_spec, mission):
    """Return a validated explicit hypothesis, or None for body-free probes."""
    if background_law_spec is None:
        if expression.uses_bodies:
            raise BackgroundModelError(
                'This candidate uses sum_b and requires model-authored '
                'background_law_spec={ax:...,ay:...}. No background dynamics '
                'are inferred automatically; explicit zero acceleration is a hypothesis too.')
        return None
    specs = validate_parameter_spec(expression.parameter_spec)
    if not isinstance(background_law_spec, dict) or set(background_law_spec) != {'ax', 'ay'}:
        raise BackgroundModelError('background_law_spec must contain exactly ax and ay expressions')
    if not all(isinstance(v, str) and v for v in background_law_spec.values()):
        raise BackgroundModelError('Background expressions must be nonempty strings')
    referenced = {n.id for text in background_law_spec.values()
                  for n in ast.walk(ast.parse(text, mode='eval')) if isinstance(n, ast.Name)}
    # A background formula may use a subset of the SAME declared parameters.
    # Expression rejects unknown symbols, including undeclared coefficients.
    selected = [p for p in specs if p['name'] in referenced]
    from .compiler import property_mapping
    mapping = property_mapping(mission)
    background = Expression(background_law_spec, selected, allowed_properties=mapping)
    initial = mission['background_initial_state']
    known = initial.get('properties', [{} for _ in initial['ids']])
    for alias in background.required_properties:
        scope, key = mapping[alias]
        if scope == 'probe' and any(key not in properties for properties in known):
            raise BackgroundModelError(
                'Background expression requires unavailable body property ' + repr(alias) +
                '; the bridge cannot invent background properties.')
    return copy.deepcopy(background_law_spec)
