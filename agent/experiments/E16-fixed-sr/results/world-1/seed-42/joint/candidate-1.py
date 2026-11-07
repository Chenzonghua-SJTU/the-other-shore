import numpy as np
import math
LAW_API_VERSION = '1.0.0'
PARAMS = {'c1': -0.02926167728830576, 'c2': 0.001635639258143594, 'c3': 0.0019555377470653543, 'c4': -0.0009162381527888161, 'c5': -0.005718053436977933}

def source_property(bodies, target, key):
    for source in bodies:
        if source['id'] == target:
            return source['properties'][key]
    raise ValueError('Public source control target is absent')

def acceleration(t, bodies, params):
    result = np.zeros((len(bodies), 2))
    for i, b in enumerate(bodies):
        if b['kind'] == 'probe':
            result[i, 0] = params['c1'] * b['pos'][0] + params['c2'] * b['pos'][0] ** 3 + params['c3'] * (b['pos'][0] ** 2 * b['pos'][1])
            result[i, 1] = params['c4'] * t + params['c5'] * b['pos'][1]
        else:
            result[i, 0] = 0
            result[i, 1] = 0
    return result
