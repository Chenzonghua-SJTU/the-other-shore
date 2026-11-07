import numpy as np
import math
LAW_API_VERSION = '1.0.0'
PARAMS = {'c1': 0.060013542508473974, 'c2': 0.0007452108973735923, 'c3': -3.37062160195168, 'c4': -0.060081617721478096}

def source_property(bodies, target, key):
    for source in bodies:
        if source['id'] == target:
            return source['properties'][key]
    raise ValueError('Public source control target is absent')

def acceleration(t, bodies, params):
    result = np.zeros((len(bodies), 2))
    for i, b in enumerate(bodies):
        if b['kind'] == 'probe':
            result[i, 0] = math.sin(params['c1'] * (source_property(bodies, 'N0', 'p1') / b['properties']['p2']) * (-b['pos'][0] - params['c2'] / (params['c3'] * math.sqrt(b['vel'][0] * b['vel'][0] + b['vel'][1] * b['vel'][1]) + math.cos(source_property(bodies, 'N0', 'p1') / b['properties']['p2'] + b['pos'][0] - math.cos(math.sin(t))))))
            result[i, 1] = b['pos'][1] * (source_property(bodies, 'N0', 'p1') / b['properties']['p2']) * params['c4']
        else:
            result[i, 0] = 0
            result[i, 1] = 0
    return result
