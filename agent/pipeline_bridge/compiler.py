"""Translate the existing restricted mathematical DSL to the official law API."""
import ast
from ..repository.shore_agent.expressions import Expression, validate_parameter_spec


def property_mapping(mission):
    controls = mission['controls']
    mapping = {k: ('probe', k) for k in controls['probe_properties']}
    mapping.update({k: ('source', k) for k in controls['source_properties']})
    if 'p2' in controls['probe_properties']:
        mapping['mass'] = ('probe', 'p2')
    if 'p1' in controls['source_properties']:
        mapping['charge'] = ('source', 'p1')
    return mapping


def compile_law(law_spec, background_law_spec, parameter_spec, mission, background_props=None):
    if background_props:
        raise ValueError('Inferred background metadata is disabled; supply explicit background_law_spec')
    specs = validate_parameter_spec(parameter_spec)
    names = {p['name'] for p in specs}
    mapping = property_mapping(mission)
    used = set()
    from .joint_expression import joint_expression
    probe_expression = joint_expression(law_spec, specs, background_law_spec, allowed_properties=mapping)
    from .background import explicit_background
    background_law_spec = explicit_background(probe_expression, background_law_spec, mission)
    for formula in (law_spec, background_law_spec):
        if formula is None:
            continue
        referenced = {n.id for value in formula.values() for n in ast.walk(ast.parse(value, mode='eval')) if isinstance(n, ast.Name)}
        selected = [p for p in specs if p['name'] in referenced]
        Expression(formula, selected, allowed_properties=mapping)
        used.update(p['name'] for p in selected)
    if used != names:
        raise ValueError('Unused parameters: ' + ', '.join(sorted(names-used)))

    class Translate(ast.NodeTransformer):
        def visit_Name(self, node):
            value = None
            if node.id in names:
                value = 'params[' + repr(node.id) + ']'
            elif node.id in ('x', 'y', 'vx', 'vy'):
                value = "b[%r][%d]" % ('vel' if node.id.startswith('v') else 'pos', 0 if node.id.endswith('x') else 1)
            elif node.id in ('bx', 'by'):
                value = "s['pos'][%d]" % (0 if node.id == 'bx' else 1)
            return ast.parse(value, mode='eval').body if value else node

        def visit_Call(self, node):
            name = node.func.id
            if name == 'q':
                scope, key = mapping[node.args[0].value]
                value = ("b['properties'][%r]" % key if scope == 'probe' else
                         "source_property(bodies, %r, %r)" % (mission['source_property_target_id'], key))
                return ast.parse(value, mode='eval').body
            if name == 'sum_b':
                inner = ast.unparse(self.visit(node.args[0]))
                return ast.parse("sum((%s) for s in bodies if s['kind'] != 'probe' and s['id'] != b['id'])" % inner, mode='eval').body
            transformed = self.generic_visit(node)
            if name != 'abs':
                transformed.func = ast.Attribute(value=ast.Name(id='math', ctx=ast.Load()), attr=name, ctx=ast.Load())
            return transformed

    def convert(expr):
        return ast.unparse(ast.fix_missing_locations(Translate().visit(ast.parse(expr, mode='eval').body)))
    params = {p['name']: p['initial'] for p in specs}
    lines = ["import numpy as np", "import math", "LAW_API_VERSION = '1.0.0'", 'PARAMS = ' + repr(params),
        'def source_property(bodies, target, key):',
        '    for source in bodies:',
        "        if source['id'] == target:",
        "            return source['properties'][key]",
        "    raise ValueError('Public source control target is absent')",
        'def acceleration(t, bodies, params):', '    result = np.zeros((len(bodies), 2))',
        '    for i, b in enumerate(bodies):', "        if b['kind'] == 'probe':",
        '            result[i, 0] = ' + convert(law_spec['ax']),
        '            result[i, 1] = ' + convert(law_spec['ay'])]
    if background_law_spec is not None:
        lines += ['        else:',
                  '            result[i, 0] = ' + convert(background_law_spec['ax']),
                  '            result[i, 1] = ' + convert(background_law_spec['ay'])]
    else:
        # The probe expression is body-free. These unused return slots satisfy
        # the fixed API; they are never a proposed model of background motion.
        lines += ['    # Background dynamics are unspecified and cannot influence this body-free probe law.']
    lines += ['    return result']
    source = '\n'.join(lines) + '\n'
    from rift.law import validate_source
    validate_source(source)
    return source
