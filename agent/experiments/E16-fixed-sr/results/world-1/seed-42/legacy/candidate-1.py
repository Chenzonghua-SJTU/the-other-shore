import numpy as np
import math
LAW_API_VERSION = '1.0.0'
PARAMS = {'c1': -0.08019102750591739, 'c2': -0.08815879983261576}

def source_property(bodies, target, key):
    for source in bodies:
        if source['id'] == target:
            return source['properties'][key]
    raise ValueError('Public source control target is absent')

def acceleration(t, bodies, params):
    result = np.zeros((len(bodies), 2))
    for i, b in enumerate(bodies):
        if b['kind'] == 'probe':
            result[i, 0] = (params['c1'] * (source_property(bodies, 'N0', 'p1') / b['properties']['p2']) * b['pos'][0] / math.sqrt(b['pos'][0] * b['pos'][0] + b['pos'][1] * b['pos'][1]) - (b['pos'][0] * b['vel'][0] + b['pos'][1] * b['vel'][1]) * (b['vel'][0] * b['vel'][0] + b['vel'][1] * b['vel'][1]) ** t) / math.sqrt(b['pos'][0] * b['pos'][0] + b['pos'][1] * b['pos'][1])
            result[i, 1] = b['pos'][1] * params['c2'] / (b['pos'][0] * b['pos'][0] + b['pos'][1] * b['pos'][1])
        else:
            result[i, 0] = 0
            result[i, 1] = 0
    return result
