import numpy as np
import math
LAW_API_VERSION = '1.0.0'
PARAMS = {'c1': 0.03447722135133797, 'c2': 0.2753009192387104, 'c3': -0.060072928489169695, 'c4': -0.00019561062385668623, 'c5': -1.7222538895608126}

def source_property(bodies, target, key):
    for source in bodies:
        if source['id'] == target:
            return source['properties'][key]
    raise ValueError('Public source control target is absent')

def acceleration(t, bodies, params):
    result = np.zeros((len(bodies), 2))
    for i, b in enumerate(bodies):
        if b['kind'] == 'probe':
            result[i, 0] = (source_property(bodies, 'N0', 'p1') / b['properties']['p2'] * b['pos'][0] + b['vel'][0] / (t / params['c1'] + b['pos'][1] - params['c2'])) * params['c3']
            result[i, 1] = source_property(bodies, 'N0', 'p1') / b['properties']['p2'] * (-params['c4'] * params['c5'] ** source_property(bodies, 'N0', 'p1') * b['pos'][0] + b['pos'][1] * params['c3'])
        else:
            result[i, 0] = 0
            result[i, 1] = 0
    return result
