import numpy as np
import math
LAW_API_VERSION = '1.0.0'
PARAMS = {'c1': 0.6380523780411709, 'c2': -0.04551692263125734, 'c3': -0.08022876889266511}

def source_property(bodies, target, key):
    for source in bodies:
        if source['id'] == target:
            return source['properties'][key]
    raise ValueError('Public source control target is absent')

def acceleration(t, bodies, params):
    result = np.zeros((len(bodies), 2))
    for i, b in enumerate(bodies):
        if b['kind'] == 'probe':
            result[i, 0] = source_property(bodies, 'N0', 'p1') / b['properties']['p2'] * math.sin(b['pos'][0] * params['c1']) * math.cos(math.cos(b['pos'][0])) * params['c2']
            result[i, 1] = b['pos'][1] * params['c3'] / (b['pos'][0] * b['pos'][0] + b['pos'][1] * b['pos'][1])
        else:
            result[i, 0] = 0
            result[i, 1] = 0
    return result
