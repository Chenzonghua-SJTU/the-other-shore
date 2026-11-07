import numpy as np
import math
LAW_API_VERSION = '1.0.0'
PARAMS = {'c1': -0.02926456741918126, 'c2': 0.001635734408887241, 'c3': 0.0019568646461318555, 'c4': 0.01084183573031158, 'c5': -0.005753177302621592}

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
            result[i, 1] = params['c4'] * b['vel'][1] + params['c5'] * b['pos'][1]
        else:
            result[i, 0] = 0
            result[i, 1] = 0
    return result
