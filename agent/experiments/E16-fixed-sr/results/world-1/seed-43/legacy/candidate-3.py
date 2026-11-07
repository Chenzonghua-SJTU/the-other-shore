import numpy as np
import math
LAW_API_VERSION = '1.0.0'
PARAMS = {'c1': -0.07938961994698061, 'c2': -0.8968753075435685, 'c3': 0.002724997842303178, 'c4': -0.09545560436464105, 'c5': 0.8558894495832929}

def source_property(bodies, target, key):
    for source in bodies:
        if source['id'] == target:
            return source['properties'][key]
    raise ValueError('Public source control target is absent')

def acceleration(t, bodies, params):
    result = np.zeros((len(bodies), 2))
    for i, b in enumerate(bodies):
        if b['kind'] == 'probe':
            result[i, 0] = params['c1'] * (source_property(bodies, 'N0', 'p1') / b['properties']['p2']) * b['pos'][0] / (b['pos'][0] * b['pos'][0] + b['pos'][1] * b['pos'][1])
            result[i, 1] = ((b['pos'][1] + b['pos'][1] * params['c2'] / ((b['pos'][0] * b['pos'][0] + b['pos'][1] * b['pos'][1]) * (b['pos'][1] - params['c3']))) * params['c4']) ** b['properties']['p2'] / (b['pos'][0] * b['pos'][0] + b['pos'][1] * b['pos'][1] + abs(b['pos'][0] * b['vel'][0] + b['pos'][1] * b['vel'][1]) + params['c5'])
        else:
            result[i, 0] = 0
            result[i, 1] = 0
    return result
