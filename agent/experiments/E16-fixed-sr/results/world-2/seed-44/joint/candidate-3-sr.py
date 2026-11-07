import numpy as np
import math
LAW_API_VERSION = '1.0.0'
PARAMS = {'c1': -0.06005568025474779, 'c2': -0.005328716238573176, 'c3': 0.11630328892940832, 'c4': -0.05994899270347818}

def source_property(bodies, target, key):
    for source in bodies:
        if source['id'] == target:
            return source['properties'][key]
    raise ValueError('Public source control target is absent')

def acceleration(t, bodies, params):
    result = np.zeros((len(bodies), 2))
    for i, b in enumerate(bodies):
        if b['kind'] == 'probe':
            result[i, 0] = source_property(bodies, 'N0', 'p1') / b['properties']['p2'] * math.sin(b['pos'][0] * params['c1'])
            result[i, 1] = (source_property(bodies, 'N0', 'p1') / b['properties']['p2'] * b['pos'][1] - params['c2'] * math.cos(b['vel'][0] * b['vel'][0] + b['vel'][1] * b['vel'][1]) / (t + params['c3'])) * params['c4']
        else:
            result[i, 0] = 0
            result[i, 1] = 0
    return result
