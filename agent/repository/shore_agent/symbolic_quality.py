"""Observation-only local differentiation and frozen-law consistency diagnostics."""
from __future__ import annotations
import ast
import time
import numpy as np


def _local_polynomial(t, values, width=9):
    result, derivative = [], []
    n=len(t); width=min(width,n)
    for i in range(n):
        start=max(0,min(i-width//2,n-width)); selected=slice(start,start+width)
        offset=t[selected]-t[i]; scale=max(float(np.max(np.abs(offset))),1e-12)
        matrix=np.vander(offset/scale,N=min(4,width),increasing=True)
        coef=np.linalg.lstsq(matrix,values[selected],rcond=None)[0]
        result.append(coef[0]); derivative.append(coef[1]/scale)
    return np.asarray(result),np.asarray(derivative)


def derivatives(probe,t,r,v):
    metadata=probe.get('observation_noise',{})
    sigma_v=float(metadata.get('velocity_sigma') or 0.)
    sigma_r=float(metadata.get('position_sigma') or 0.)
    if not np.isfinite([sigma_v,sigma_r]).all() or min(sigma_v,sigma_r)<0: raise ValueError('invalid_noise_metadata')
    smooth_v,fine=(v,np.gradient(v,t,axis=0)) if sigma_v==0 else _local_polynomial(t,v)
    smooth_r=r if sigma_r==0 else _local_polynomial(t,r)[0]
    uncertainty=np.zeros_like(fine)
    if len(t)>=5:
        left,right=t[2:-2]-t[:-4],t[4:]-t[2:-2]
        wide=(-right/(left*(left+right)))[:,None]*smooth_v[:-4]
        wide+=((right-left)/(left*right))[:,None]*smooth_v[2:-2]
        wide+=(left/(right*(left+right)))[:,None]*smooth_v[4:]
        uncertainty[2:-2]=(wide-fine[2:-2])/3.
        uncertainty[1]=uncertainty[2]; uncertainty[-2]=uncertainty[-3]
    # Local resolution sensitivity, not a hidden-force error estimate. Retain
    # every sample and normalize per trajectory after downweighting.
    signal=np.linalg.norm(fine[1:-1],axis=1)
    noise_floor=sigma_v/max(float(np.median(np.diff(t))),1e-12)
    scale=.02*max(float(np.median(signal)),1e-12)+noise_floor+1e-12
    discrepancy=np.linalg.norm(uncertainty[1:-1],axis=1)
    quality=np.maximum(1e-6,1/(1+(discrepancy/scale)**2))
    details={'probe_id':probe['probe_id'],'method':'local_cubic_velocity' if sigma_v else 'centered_velocity_difference',
             'position_smoothed':bool(sigma_r),'velocity_noise_sigma':sigma_v,
             'position_noise_sigma':sigma_r,'samples':len(quality),
             'downweighted_samples':int(np.sum(quality<.5)),
             'low_reliability_times':t[1:-1][quality<.1].tolist(),
             'minimum_relative_weight':float(quality.min()),
             'scope':'Resolution sensitivity only; no samples removed and no truth used.'}
    return smooth_r[1:-1],smooth_v[1:-1],fine[1:-1],quality,uncertainty[1:-1]**2,details


def consistency(law_spec,parameters,probes,*,deadline=float('inf')):
    """Integral residual on observed paths, without derivative targets or new rollout.

    Quadrature and observation errors remain; this is discovery feedback and
    cannot certify a law or masquerade as fresh independent validation.
    """
    from .symbolic_regression import _evaluate_array
    rows=[]
    for probe in probes:
        if time.monotonic()>=deadline: break
        t,r,v=(np.asarray(probe[k],dtype=float) for k in ('times','r_obs','v_obs'))
        context=dict(parameters,t=t,x=r[:,0],y=r[:,1],vx=v[:,0],vy=v[:,1])
        context.update(probe.get('initial_conditions',{}).get('properties',{}))
        row={'probe_id':probe['probe_id']}
        try:
            acceleration=np.column_stack([np.broadcast_to(_evaluate_array(law_spec[a],context),t.shape) for a in ('ax','ay')])
            increment=(acceleration[:-1]+acceleration[1:])*.5*np.diff(t)[:,None]
            observed=np.diff(v,axis=0)
            residual=increment-observed
            # Windowed accumulated errors expose persistent drift without using
            # every long trajectory's length as an implicit weighting factor.
            cumulative=np.cumsum(residual,axis=0)
            error=float(np.mean(np.sum(residual**2,axis=1)))
            scale=float(np.mean(np.sum(observed**2,axis=1)))
            row.update(status='checked',velocity_increment_rmse=error**.5,
                       relative_increment_rmse=(error/max(scale,1e-20))**.5,
                       cumulative_velocity_residual_rmse=float(np.sqrt(np.mean(np.sum(cumulative**2,axis=1)))))
            crossings=0
            for source in law_spec.values():
                for node in ast.walk(ast.parse(source,mode='eval')):
                    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div):
                        denominator=np.broadcast_to(_evaluate_array(ast.unparse(node.right),context),t.shape)
                        crossings+=int(np.sum(denominator[:-1]*denominator[1:]<0))
            row['denominator_sign_changes']=crossings
        except (ValueError,TypeError,OverflowError,ZeroDivisionError):
            row.update(status='domain_error',reason='Candidate is invalid on an acquired state.')
        rows.append(row)
    good=[r for r in rows if r['status']=='checked']
    return {'scope':'Frozen formula on acquired observed paths; trapezoidal velocity increments, not hidden truth or independent validation.',
            'per_probe':rows,'checked_probe_ids':[r['probe_id'] for r in rows],
            'complete':len(rows)==len(probes),'domain_error_probe_ids':[r['probe_id'] for r in rows if r['status']!='checked'],
            'denominator_sign_change_probe_ids':[r['probe_id'] for r in good if r['denominator_sign_changes']],
            'velocity_increment_rmse':float(np.sqrt(np.mean([r['velocity_increment_rmse']**2 for r in good]))) if good else None,
            'relative_increment_rmse':float(np.sqrt(np.mean([r['relative_increment_rmse']**2 for r in good]))) if good else None,
            'action':'A contradiction remains evidence even when a probe is omitted from train_probe_ids; acquire denser controls if quadrature resolution is insufficient.'}
