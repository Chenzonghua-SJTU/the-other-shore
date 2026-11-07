import numpy as np
import math
LAW_API_VERSION = '1.0.0'
PARAMS = {'c1': -0.06002585963414175, 'c2': 0.00046099423070216646, 'c3': 0.04711936215852369, 'c4': -0.0599835762955261, 'c5': 0.00018556431552842526}

def source_property(bodies, target, key):
    for source in bodies:
        if source['id'] == target:
            return source['properties'][key]
    raise ValueError('Public source control target is absent')

def acceleration(t, bodies, params):
    result = np.zeros((len(bodies), 2))
    for i, b in enumerate(bodies):
        if b['kind'] == 'probe':
            result[i, 0] = b['pos'][0] * params['c1'] * (source_property(bodies, 'N0', 'p1') / b['properties']['p2']) - -1 * params['c2'] / (t / math.cos(b['pos'][0]) + math.sin(b['pos'][0] + b['pos'][0] + params['c3']))
            result[i, 1] = source_property(bodies, 'N0', 'p1') / b['properties']['p2'] * b['pos'][1] * params['c4'] - 1 * params['c5']
        else:
            result[i, 0] = 0
            result[i, 1] = 0
    return result
