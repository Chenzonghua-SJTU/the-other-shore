"""Validate the declared parameter union without changing either physical formula."""
import ast
from contextvars import ContextVar
from ..repository.shore_agent.expressions import Expression, ExpressionError, validate_parameter_spec, ALLOWED_PROPERTIES

background_context = ContextVar('explicit_background_formula', default=None)

def references(formula):
    try:
        return {node.id for text in formula.values() for node in ast.walk(ast.parse(text, mode='eval')) if isinstance(node, ast.Name)}
    except (SyntaxError, TypeError, AttributeError) as exc:
        raise ExpressionError('formula is not valid DSL syntax') from exc

def joint_expression(law_spec, parameter_spec, background=None, allowed_properties=ALLOWED_PROPERTIES):
    specs = validate_parameter_spec(parameter_spec)
    if background is None:
        return Expression(law_spec, specs, allowed_properties)
    names = {p['name'] for p in specs}
    probe_names = references(law_spec)
    background_names = references(background)
    probe = Expression(law_spec, [p for p in specs if p['name'] in probe_names], allowed_properties)
    Expression(background, [p for p in specs if p['name'] in background_names], allowed_properties)
    unused = names - probe_names - background_names
    if unused:
        raise ExpressionError('unused declared parameters: ' + ', '.join(sorted(unused)))
    # The optimizer and official rollout must see every explicitly declared
    # coefficient, including coefficients used only by background dynamics.
    probe.parameter_spec = specs
    probe.parameter_names = names
    probe.used_parameters = names
    return probe

def scoped_expression(law_spec, parameter_spec, allowed_properties=ALLOWED_PROPERTIES):
    return joint_expression(law_spec, parameter_spec, background_context.get(), allowed_properties)
