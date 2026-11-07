import numpy as np
import math
LAW_API_VERSION = '1.0.0'
PARAMS = {'c1': -0.027913743659728797, 'c2': 0.0016174533068779157, 'c3': -0.00555212556381769, 'c4': -0.4717786276436149, 'c5': -0.001474072320266323}

def source_property(bodies, target, key):
    for source in bodies:
        if source['id'] == target:
            return source['properties'][key]
    raise ValueError('Public source control target is absent')

def acceleration(t, bodies, params):
    result = np.zeros((len(bodies), 2))
    for i, b in enumerate(bodies):
        if b['kind'] == 'probe':
            result[i, 0] = params['c1'] * b['pos'][0] + params['c2'] * b['pos'][0] ** 3
            result[i, 1] = params['c3'] * b['pos'][1] + params['c4'] * (b['vel'][0] * b['vel'][1]) + params['c5'] * (b['pos'][0] * b['pos'][1])
        else:
            result[i, 0] = 0
            result[i, 1] = 0
    return result
