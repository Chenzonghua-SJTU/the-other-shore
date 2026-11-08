"""Independent reference law fixtures for validating the public training package.

These sources are correct answers, not discovery Agents. Never expose this
module or its output to a policy when measuring unknown-law discovery. The
formulas below are independent scalar implementations of the simulator.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

from .worlds import PARAMETERS, SINGLE_CENTER, WORLDS


FIELDS = {
    '1': '''
        radius = math.hypot(x, y)
        if radius == 0.: return (0., 0.)
        scale = -p['G'] * strength / (4 * math.pi * inertia * math.sqrt(radius*radius + .05*.05) * radius)
        return (scale*x, scale*y)
    ''',
    '2': '''
        scale = -p['k'] * strength / inertia
        return (scale*x, scale*y)
    ''',
    '3': '''
        c, s = math.cos(p['angle']), math.sin(p['angle'])
        u, v = c*x+s*y, -s*x+c*y
        return (-c*p['kx']*u+s*p['ky']*v, -s*p['kx']*u-c*p['ky']*v)
    ''',
    '4': '''
        radius = math.hypot(x, y)
        if radius == 0.: return (0., 0.)
        rho = math.sqrt(radius*radius + .05*.05)
        scale = -p['G'] * strength * math.exp(-rho/p['lambda']) / (2 * math.pi * inertia * rho * radius)
        return (scale*x, scale*y)
    ''',
    '5': '''
        return (-p['a']*x-p['c']*x**3-p['d']*x*y*y,
                -p['b']*y-p['c']*y**3-p['d']*y*x*x)
    ''',
    '6': '''
        theta = math.pi*t/p['period']+p['phase']
        c, s = math.cos(theta), math.sin(theta)
        u, v = c*x+s*y, -s*x+c*y
        return (-c*p['kx']*u+s*p['ky']*v, -s*p['kx']*u-c*p['ky']*v)
    ''',
    '7': '''
        scale = -p['G']*strength/inertia / (x*x+y*y+p['s']**2)**1.5
        return (scale*x, scale*y)
    ''',
    '8': '''
        scale = -p['k']*strength/inertia / math.sqrt(1+(x*x+y*y)/p['L']**2)
        return (scale*x, scale*y)
    ''',
    '9': '''
        scale = -(strength/inertia)*(p['k']+p['c']*(x*x+y*y))
        return (scale*x, scale*y)
    ''',
    '10': '''
        return (p['a']*x-p['b']*x**3, -p['ky']*y)
    ''',
    '11': '''
        return (-p['k']*x-p['A']*p['q']*math.sin(p['q']*x),
                -p['k']*y-p['A']*p['q']*math.sin(p['q']*y))
    ''',
    '12': '''
        scale = -(strength/inertia)*p['A']*math.exp(-(x*x+y*y)/(2*p['s']**2))
        return (scale*x, scale*y)
    ''',
    '13': '''
        scale = -p['k']*(x*x+y*y-p['radius']**2)
        return (scale*x, scale*y)
    ''',
    '14': '''
        ax, ay = -p['k']*x, -p['k']*y
        for source, strength in zip(sources, (p['G1'], p['G2'])):
            dx, dy = x-source[0], y-source[1]
            scale = -strength/(dx*dx+dy*dy+p['s']**2)**1.5
            ax, ay = ax+scale*dx, ay+scale*dy
        return (ax, ay)
    ''',
    '15': '''
        dx, dy = x-p['cx'], y-p['cy']
        scale = -p['G']/(dx*dx+dy*dy+p['s']**2)**1.5
        return (-p['k']*x+scale*dx, -p['k']*y+scale*dy)
    ''',
    '16': '''
        ax, ay = -p['k']*x, -p['k']*y
        for cx, strength in ((-p['d'], p['G1']), (p['d'], p['G2'])):
            dx = x-cx
            scale = -strength/(dx*dx+y*y+p['s']**2)**1.5
            ax, ay = ax+scale*dx, ay+scale*y
        return (ax, ay)
    ''',
    '17': '''
        scale = -p['k']*(1+p['amplitude']*math.sin(2*math.pi*t/p['period']+p['phase']))
        return (scale*x, scale*y)
    ''',
    '18': '''
        theta = 2*math.pi*t/p['period']
        return (-p['k']*x+p['A']*math.cos(theta), -p['k']*y+p['A']*math.sin(theta))
    ''',
    '19': '''
        phase = p['q']*x-2*math.pi*t/p['period']+p['phase']
        return (-p['k']*x+p['A']*p['q']*math.sin(phase), -p['k']*y)
    ''',
    '20': '''
        scale = -p['k']*(1+p['amplitude']*math.sin(2*math.pi*t/p['period']))
        return (scale*(x-p['cx']), scale*(y-p['cy']))
    ''',
}


def oracle_source(world):
    world = str(world)
    if world not in WORLDS:
        raise ValueError('World must be one of the public training IDs.')
    code = textwrap.indent(textwrap.dedent(FIELDS[world]).strip(), '    ')
    # Fixed visible sources/markers do not move. Other background bodies are
    # passive tracers under exactly the same field as probes. The hidden-source
    # coordinates are fitted scalars in PARAMS; analytic forces need no HIDDEN
    # dynamic state and introduce no uncounted adjustable coordinates.
    fixed = ('N0', 'N1') if world == '14' else ('N0',)
    source_setup = ("sources = [next(b['pos'] for b in bodies if b['id'] == name) for name in ('N0', 'N1')]"
                    if world == '14' else 'sources = []')
    strength = ("next(b['properties'].get('p1', 1.) for b in bodies if b['id'] == 'N0')"
                if world in SINGLE_CENTER else '1.')
    inertia = "body['properties'].get('p2', 1.)" if world in SINGLE_CENTER else '1.'
    return f'''"""Public reference answer for world {world}; infrastructure validation only."""
import math
import numpy as np

LAW_API_VERSION = '1.0.0'
PARAMS = {PARAMETERS[world]!r}
BODY_PROPS = {{}}
HIDDEN = []


def _field(t, x, y, p, strength, inertia, sources):
{code}


def acceleration(t, bodies, params):
    out = np.zeros((len(bodies), 2), dtype=np.float64)
    {source_setup}
    strength = {strength}
    for i, body in enumerate(bodies):
        if body['kind'] == 'background' and body['id'] in {fixed!r}:
            continue
        x, y = body['pos']
        out[i] = _field(t, x, y, params, strength, {inertia}, sources)
    return out
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', choices=WORLDS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as out:
        out.write(oracle_source(args.world))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
