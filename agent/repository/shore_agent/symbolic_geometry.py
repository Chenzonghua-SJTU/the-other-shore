"""Generic coordinate features and a data-tested vector projection hypothesis.

No simulator, world catalogue, force template, or fitted physical constant.
Aliases are expanded back to ordinary state expressions before tool export.
"""
from __future__ import annotations

import ast
import copy
import numpy as np

ALIASES = {
    'radius2': 'x*x+y*y', 'radius': 'sqrt(x*x+y*y)',
    'speed2': 'vx*vx+vy*vy', 'radial_dot': 'x*vx+y*vy',
    'property_ratio': 'charge/mass',
}


def feature_table(dataset):
    names = list(dataset['feature_names'])
    values = [dataset['X'][:, i] for i in range(len(names))]
    state = dataset['states']
    x, y, vx, vy = (state[:, i] for i in (1, 2, 3, 4))
    options = {'radius2': x*x+y*y, 'radius': np.hypot(x, y),
               'speed2': vx*vx+vy*vy, 'radial_dot': x*vx+y*vy}
    if 'mass' in names and 'charge' in names:
        mass, charge = (dataset['X'][:, names.index(k)] for k in ('mass', 'charge'))
        if np.all(np.abs(mass) > 1e-12): options['property_ratio'] = charge/mass
    for key, value in options.items():
        dependencies = {'radius2': ('x','y'), 'radius': ('x','y'),
                        'speed2': ('vx','vy'), 'radial_dot': ('x','y','vx','vy'),
                        'property_ratio': ('mass','charge')}[key]
        # A dropped constant state variable is not silently reintroduced by an alias.
        if (all(k in names for k in dependencies) and np.isfinite(value).all()
                and np.max(np.abs(value)) <= 1e12 and np.ptp(value) > 1e-12):
            names.append(key); values.append(value)
    return (np.column_stack(values) if values else np.empty((len(state), 0))), names


def expand_aliases(source):
    class Expand(ast.NodeTransformer):
        def visit_Name(self, node):
            return copy.deepcopy(ast.parse(ALIASES[node.id], mode='eval').body) if node.id in ALIASES else node
    tree = Expand().visit(ast.parse(source, mode='eval'))
    return ast.unparse(ast.fix_missing_locations(tree))


def radial_projection(dataset, *, tolerance=.04):
    """Propose a scalar multiplier only when acquired vectors support collinearity.

    This is a hypothesis about these observations, not a symmetry assumption.
    General component searches remain available, including direction dependence.
    """
    if not {'x', 'y'} <= set(dataset['feature_names']):
        return None, {'supported': False, 'reason': 'both_coordinate_features_required'}
    position = dataset['states'][:, 1:3]
    acceleration = dataset['y']
    r2 = np.sum(position*position, axis=1)
    if np.any(r2 < 1e-10) or len(position) < 6:
        return None, {'supported': False, 'reason': 'insufficient_nonzero_position_states'}
    weights = dataset['weights']
    signal = float(np.sum(weights[:, None]*acceleration**2))
    if signal < 1e-20:
        return None, {'supported': False, 'reason': 'acceleration_signal_too_small'}
    scalar = np.sum(position*acceleration, axis=1)/r2
    perpendicular = acceleration-scalar[:, None]*position
    error = float(np.sqrt(np.sum(weights[:, None]*perpendicular**2)/signal))
    angular = position/np.sqrt(r2)[:, None]
    eigenvalues = np.linalg.eigvalsh(angular.T@(weights[:, None]*angular))
    # A small *relative* transverse force is still a resolved contradiction
    # when velocity differences are precise. A fixed 4% allowance incorrectly
    # gave weakly noncentral observations most of the radial search budget.
    # Trace of the two-axis resolution estimate is rotation invariant. This is
    # a stencil/noise sensitivity heuristic, not a statistical confidence bound.
    relative_resolution = None
    resolution = np.asarray(dataset.get('derivative_resolution_mse', []), dtype=float)
    if resolution.shape == (2,) and np.isfinite(resolution).all() and np.all(resolution >= 0):
        relative_resolution = float(np.sqrt(np.sum(resolution)/signal))
    effective_tolerance = (min(tolerance, max(1e-8, 4*relative_resolution))
                           if relative_resolution is not None else tolerance)
    supported = error <= effective_tolerance and eigenvalues[0] > .02
    diagnostics = {'supported': bool(supported), 'relative_transverse_residual': error,
                   'relative_derivative_resolution': relative_resolution,
                   'effective_transverse_tolerance': effective_tolerance,
                   'tolerance_rule': 'min(relative cap, four times observed stencil/noise sensitivity); not a confidence interval',
                   'direction_coverage_min_eigenvalue': float(eigenvalues[0]),
                   'hypothesis': 'acceleration equals an unknown scalar response times position',
                   'scope': 'Acquired-data hypothesis only; not proof of isotropy or an origin force source.'}
    if not supported: return None, diagnostics
    # Scalar loss weighted by r² equals vector acceleration loss for this hypothesis.
    projected_weights = weights*r2
    projected_weights /= projected_weights.sum()
    return {'target': scalar, 'weights': projected_weights}, diagnostics


def radial_log_rows(dataset,projected,*,deadline):
    """Generic sign-preserving log-response regression, conditional on collinearity.

    Arbitrary subsets of observed scalar coordinates/properties are tested. No
    exponent, range, coefficient, world id or physical force family is supplied.
    Ordinary Cartesian/PySR searches remain available alongside this hypothesis.
    """
    import itertools,time
    if projected is None: return [[],[]],{'enabled':False}
    response=projected['target']
    if not (np.all(response>0) or np.all(response<0)):
        return [[],[]],{'enabled':False,'reason':'response_changes_sign'}
    sign=1 if response[0]>0 else -1
    radius=np.linalg.norm(dataset['states'][:,1:3],axis=1)
    features=[('sqrt(x*x+y*y)',radius),('log(sqrt(x*x+y*y))',np.log(radius))]
    for name in ('t','mass','charge'):
        if name not in dataset['feature_names']: continue
        values=dataset['X'][:,dataset['feature_names'].index(name)]
        features.append((name,values))
        if name!='t' and np.all(values>0): features.append((f'log({name})',np.log(values)))
    # Fit relative log-response with geometric/discretization weights. Multiplying
    # by response**2 would let unresolved near-zero-radius values dominate again.
    weights=projected['weights'].copy()
    weights/=weights.sum(); root=np.sqrt(weights)
    target=np.log(np.abs(response))
    rows=[[],[]]; tried=0
    for size in range(1,min(4,len(features))+1):
        for subset in itertools.combinations(range(len(features)),size):
            if time.monotonic()>=deadline: break
            design=np.column_stack([np.ones(len(response))]+[features[j][1] for j in subset])
            coef,_,rank,_=np.linalg.lstsq(design*root[:,None],target*root,rcond=None)
            if rank!=size+1 or not np.isfinite(coef).all(): continue
            tried+=1
            body=repr(float(coef[0]))+' + '+' + '.join(f'({float(c)!r})*({features[j][0]})' for c,j in zip(coef[1:],subset))
            scalar=f'{sign}*exp({body})'
            for axis,name in enumerate(('x','y')):
                rows[axis].append({'source':f'({scalar})*{name}','complexity':6+5*size,
                                  'origin':'transformed_linear_regression','search_representation':'sign_checked_radial_log_response'})
    return rows,{'enabled':True,'models_fitted':tried,'target':'log(abs(observed_radial_scalar))',
                 'features':[f[0] for f in features],'scope':'Data-supported sign/collinearity hypothesis; coefficients fitted only to discovery observations.'}
