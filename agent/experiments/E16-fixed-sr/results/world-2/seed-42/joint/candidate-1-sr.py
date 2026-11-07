import numpy as np
import math
LAW_API_VERSION = '1.0.0'
PARAMS = {'c1': -6.863262788211006e-05, 'c2': 0.09918636598944336, 'c3': 0.06012014246579006, 'c4': -0.06002071224098931, 'c5': 0.00019380890967919816}

def source_property(bodies, target, key):
    for source in bodies:
        if source['id'] == target:
            return source['properties'][key]
    raise ValueError('Public source control target is absent')

def acceleration(t, bodies, params):
    result = np.zeros((len(bodies), 2))
    for i, b in enumerate(bodies):
        if b['kind'] == 'probe':
            result[i, 0] = source_property(bodies, 'N0', 'p1') * math.sin((b['pos'][0] * b['pos'][0] + b['pos'][1] * b['pos'][1]) * params['c1'] / (-t - params['c2']) - params['c3'] * b['pos'][0] / b['properties']['p2'])
            result[i, 1] = b['pos'][1] * math.sin(source_property(bodies, 'N0', 'p1') / b['properties']['p2'] * params['c4']) - 1 * params['c5']
        else:
            result[i, 0] = 0
            result[i, 1] = 0
    return result
