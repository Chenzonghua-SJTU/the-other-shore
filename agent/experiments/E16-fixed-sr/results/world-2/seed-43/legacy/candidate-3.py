import numpy as np
import math
LAW_API_VERSION = '1.0.0'
PARAMS = {'c1': -0.0005773183107032392, 'c2': 0.018238656960855675, 'c3': -0.06006083331179296, 'c4': -0.012989551062175201, 'c5': -0.060086491913561586}

def source_property(bodies, target, key):
    for source in bodies:
        if source['id'] == target:
            return source['properties'][key]
    raise ValueError('Public source control target is absent')

def acceleration(t, bodies, params):
    result = np.zeros((len(bodies), 2))
    for i, b in enumerate(bodies):
        if b['kind'] == 'probe':
            result[i, 0] = (source_property(bodies, 'N0', 'p1') / b['properties']['p2'] * params['c1'] / math.sin(b['pos'][0] * b['vel'][0] + b['pos'][1] * b['vel'][1] + (b['pos'][0] * b['vel'][0] + b['pos'][1] * b['vel'][1]) + t + params['c2']) + b['pos'][0]) * (source_property(bodies, 'N0', 'p1') / b['properties']['p2']) * params['c3']
            result[i, 1] = source_property(bodies, 'N0', 'p1') / b['properties']['p2'] * (b['pos'][1] - math.sin(math.sin((b['pos'][0] * b['pos'][0] + b['pos'][1] * b['pos'][1]) * math.sin(t + math.sin(b['pos'][0]))) * params['c4'])) * params['c5']
        else:
            result[i, 0] = 0
            result[i, 1] = 0
    return result
