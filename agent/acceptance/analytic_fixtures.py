"""Independent analytic trajectories. Truth stays in the test harness only."""
from __future__ import annotations
import copy
import numpy as np

SEEDS = (301, 302, 303, 304, 305)
NOISE_LEVELS = ((0.004, 0.0007), (0.04, 0.003))
FAMILIES = ('polynomial', 'mixed_frequency', 'chirp', 'localized_pulse')


def truth(family, u):
    u = np.asarray(u, dtype=float)
    if family == 'polynomial':
        r = np.column_stack((.7 + .21*u - .024*u*u + .0017*u**3,
                             -.3 + .16*u + .011*u*u - .0008*u**3))
        v = np.column_stack((.21 - .048*u + .0051*u*u,
                             .16 + .022*u - .0024*u*u))
        a = np.column_stack((-.048 + .0102*u, .022 - .0048*u))
    elif family == 'mixed_frequency':
        w, z = .63, 2.17
        r = np.column_stack((.9*np.sin(w*u) + .11*np.sin(z*u) + .07*u,
                             .6*np.cos(.83*u) + .08*np.sin(1.71*u)))
        v = np.column_stack((.9*w*np.cos(w*u) + .11*z*np.cos(z*u) + .07,
                             -.6*.83*np.sin(.83*u) + .08*1.71*np.cos(1.71*u)))
        a = np.column_stack((-.9*w*w*np.sin(w*u) - .11*z*z*np.sin(z*u),
                             -.6*.83**2*np.cos(.83*u) - .08*1.71**2*np.sin(1.71*u)))
    elif family == 'chirp':
        p, dp, ddp = .31*u + .097*u*u, .31 + .194*u, .194
        q, dq, ddq = .47*u + .061*u*u, .47 + .122*u, .122
        r = np.column_stack((.75*np.sin(p), .55*np.cos(q) + .04*u))
        v = np.column_stack((.75*np.cos(p)*dp, -.55*np.sin(q)*dq + .04))
        a = np.column_stack((.75*(-np.sin(p)*dp*dp + np.cos(p)*ddp),
                             -.55*(np.cos(q)*dq*dq + np.sin(q)*ddq)))
    elif family == 'localized_pulse':
        c, width = 4.1, .52
        z=(u-c)/width; g=np.exp(-z*z); dg=-2*z*g/width
        ddg=(4*z*z-2)*g/(width*width)
        r=np.column_stack((.14*u + .13*g, -.06*u + .09*g + .004*u*u))
        v=np.column_stack((.14 + .13*dg, -.06 + .09*dg + .008*u))
        a=np.column_stack((.13*ddg, .09*ddg + .008))
    else:
        raise ValueError(family)
    return r,v,a


def case(family, seed, sigma_r, sigma_v, *, irregular=False, n=50, t0=1.7):
    rng=np.random.default_rng(seed)
    u=np.linspace(0.,9.3,n)
    if irregular and n>2:
        u[1:-1] += rng.uniform(-.22,.22,n-2)*(u[1]-u[0])
    r,v,a=truth(family,u)
    noisy_r=r+rng.normal(0.,sigma_r,r.shape)
    noisy_v=v+rng.normal(0.,sigma_v,v.shape)
    probe={'probe_id':f'acceptance_{family}_{seed}',
           'times':(u+t0).tolist(),'r_obs':noisy_r.tolist(),'v_obs':noisy_v.tolist(),
           'initial_conditions':{'r0':r[0].tolist(),'v0':v[0].tolist(),'t0':float(t0),
                                 'properties':{'mass':1.3,'charge':-.8}},
           'observation_noise':{'position_sigma':float(sigma_r),'velocity_sigma':float(sigma_v)}}
    # Keep truth and identifiers describing the family outside the tool input.
    probe['probe_id']=f'probe_{seed:04d}'
    return {'family':family,'seed':seed,'irregular':irregular,'noise':[sigma_r,sigma_v],
            'probe':probe,'truth':{'r':r,'v':v,'a':a}}


def core_suite():
    for family in FAMILIES:
        for seed in SEEDS:
            for sigma_r,sigma_v in NOISE_LEVELS:
                for irregular in (False,True):
                    yield case(family,seed,sigma_r,sigma_v,irregular=irregular)


def rms(array):
    return float(np.sqrt(np.mean(np.sum(np.asarray(array)**2,axis=-1))))
