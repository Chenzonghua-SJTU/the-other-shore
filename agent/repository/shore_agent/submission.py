"""Generate standalone deterministic law code from a validated AST interpreter."""
import importlib.util
import json
import py_compile
import uuid
from pathlib import Path
import numpy as np
from . import expressions


def export_submission(directory, law, claims, probes, physical_explanation):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    parameters = law['fitted_parameters']
    if len(parameters) > 5 or any(isinstance(v, bool) or not np.isfinite(v) for v in parameters.values()):
        raise ValueError('Parameters must be at most five finite scalars')
    # Copy only our vetted, standalone DSL interpreter. No model-authored Python executes.
    source = Path(expressions.__file__).read_text(encoding='utf-8')
    source += '\n\n# Frozen submission metadata; no environment or archive access.\n'
    source += 'PARAMS = ' + repr(parameters) + '\n'
    source += 'LAW_SPEC = ' + repr(law['law_spec']) + '\n'
    source += 'PARAMETER_SPEC = ' + repr(law['parameter_spec']) + '\n'
    source += '''
_LAW = Expression(LAW_SPEC, PARAMETER_SPEC)

def acceleration(t, bodies, params=None):
    """Local contract: bodies={r:[x,y], v:[vx,vy], visible:[...], properties:{...}}.
    visible supplies the visible bodies' instantaneous positions at absolute t.
    Force-law parameters are frozen; externally supplied positions may change.
    The formal competition state encoding must be adapted when published.
    """
    return _LAW.acceleration(t, bodies['r'], bodies['v'], bodies['visible'],
                             bodies.get('properties', {}), PARAMS if params is None else params).tolist()
'''
    path = directory / 'law.py'
    path.write_text(source, encoding='utf-8')
    (directory / 'claims.json').write_text(json.dumps(claims, indent=2), encoding='utf-8')
    (directory / 'law.json').write_text(json.dumps(law, indent=2), encoding='utf-8')
    # Explanatory prose is an artifact, never part of the executable law.
    (directory / 'physical_explanation.txt').write_text(physical_explanation, encoding='utf-8')
    py_compile.compile(str(path), cfile=str(directory / 'law.pyc'), doraise=True)
    spec = importlib.util.spec_from_file_location('submitted_' + uuid.uuid4().hex, path)
    module = importlib.util.module_from_spec(spec)
    # Generated code is trusted interpreter code plus literal metadata, not arbitrary source from a model.
    import sys
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        points = 0
        body_input = law.get('body_context', law.get('known_bodies', []))
        for p in probes:
            for t, r, v in zip(p['times'], p['r_obs'], p['v_obs']):
                visible = expressions.body_positions(body_input, t) if module._LAW.uses_bodies else []
                state = {'r': r, 'v': v, 'visible': visible,
                         'properties': p['initial_conditions']['properties']}
                actual = np.asarray(module.acceleration(t, state))
                if actual.shape != (2,) or not np.isfinite(actual).all():
                    raise ValueError('Invalid acceleration output')
                points += 1
    finally:
        sys.modules.pop(spec.name, None)
    return {'format': True, 'finite_parameters': True, 'executable': True,
            'finite_observed_state_outputs': points, 'parameter_count': len(parameters),
            'body_input_protocol': law.get('body_input_protocol', 'frozen_static_observed_positions'),
            'state_encoding': 'local adapter, not a claim of official API compatibility'}
