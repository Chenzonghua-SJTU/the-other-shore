import numpy as np
import math
LAW_API_VERSION = '1.0.0'
PARAMS = {'c1': -0.0600933376440268, 'c2': 0.0007090357164786891, 'c3': -0.015262681987720851, 'c4': 1.0412265763838633, 'c5': -0.06010435079877045}

def source_property(bodies, target, key):
    for source in bodies:
        if source['id'] == target:
            return source['properties'][key]
    raise ValueError('Public source control target is absent')

def acceleration(t, bodies, params):
    result = np.zeros((len(bodies), 2))
    for i, b in enumerate(bodies):
        if b['kind'] == 'probe':
            result[i, 0] = source_property(bodies, 'N0', 'p1') / b['properties']['p2'] * (b['pos'][0] * params['c1'] + params['c2'] / (t + math.sin(math.sqrt(b['pos'][0] * b['pos'][0] + b['pos'][1] * b['pos'][1]))))
            result[i, 1] = (b['pos'][1] - params['c3'] * math.sin((b['pos'][0] * b['pos'][0] + b['pos'][1] * b['pos'][1]) * math.sin(math.sin(t + math.sin(b['pos'][0]))) - math.cos(b['pos'][1] + params['c4']))) * math.sin(source_property(bodies, 'N0', 'p1') / b['properties']['p2'] * params['c5'])
        else:
            result[i, 0] = 0
            result[i, 1] = 0
    return result
