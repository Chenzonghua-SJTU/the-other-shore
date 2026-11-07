"""Autonomous background hypotheses inferred only from acquired public observations.

No world IDs, catalog access, future observation tables, or alternate integrator.
Centers are per-object geometric metadata (BODY_PROPS), never shared law PARAMS.
"""
import hashlib,json
import numpy as np
from scipy.optimize import least_squares

class BackgroundModelError(ValueError):pass

def circle_positions(times,p0,v0,center):
    delta=np.asarray(p0)-center;radius2=float(delta@delta)
    if radius2<1e-16:raise BackgroundModelError('Degenerate circular center')
    omega=float((delta[0]*v0[1]-delta[1]*v0[0])/radius2)
    if abs(omega)<1e-14:raise BackgroundModelError('Degenerate angular rate')
    t=np.asarray(times);angle=omega*t
    return np.asarray(p0)[None,:]+(np.cos(angle)-1)[:,None]*delta[None,:]+(t*np.sinc(angle/np.pi))[:,None]*np.asarray(v0)[None,:]

def infer(mission,context,sigma):
    initial=mission['background_initial_state'];ids=initial['ids'];p0=np.asarray(initial['positions'],float);v0=np.asarray(initial['velocities'],float)
    if not ids:return {},[]
    if not isinstance(context,dict) or not context.get('times'):
        if np.linalg.norm(v0)>1e-12:raise BackgroundModelError('Acquired background observations are required for moving objects')
        return {},[]
    times=np.asarray(context['times'],float);positions=np.asarray(context['positions'],float);observed=context.get('body_ids',ids)
    if positions.shape!=(len(times),len(observed),2):raise BackgroundModelError('Unsupported acquired background record shape')
    sigma=float(sigma or 0.);props={};audit=[]
    stationary_centers=[p0[i] for i in range(len(ids)) if np.linalg.norm(v0[i])<1e-12]
    for i,ident in enumerate(ids):
        if ident not in observed:raise BackgroundModelError('Background ID missing from acquired observations')
        y=positions[:,observed.index(ident),:];valid=np.isfinite(y).all(axis=1)&np.isfinite(times);t=times[valid];y=y[valid]
        if len(t)<3:raise BackgroundModelError('Too few acquired background positions')
        residual=lambda pred:float(np.sqrt(np.mean(np.sum((pred-y)**2,axis=1))))
        linear=p0[i]+t[:,None]*v0[i];choices=[{'mode':'linear','center':None,'rmse':residual(linear),'fitted_geometry':0}]
        speed=float(np.linalg.norm(v0[i]));extent=max(1.,float(np.linalg.norm(p0[i])),float(np.linalg.norm(y,axis=1).max()))
        if speed>1e-12:
            n=np.array([-v0[i,1],v0[i,0]])/speed
            centers=[]
            for c in stationary_centers:
                delta=p0[i]-c
                if np.linalg.norm(delta)>1e-7 and abs(float(delta@v0[i]))<=1e-8*max(1.,np.linalg.norm(delta)*speed):centers.append((c,0))
            d=y-p0[i];denom=2*(d@n);good=np.abs(denom)>1e-10
            if np.count_nonzero(good)>=2:
                distance=float(np.sum(denom[good]*np.sum(d[good]**2,axis=1))/np.sum(denom[good]**2))
                if abs(distance)>1e-5 and abs(distance)<extent*1e5:
                    sign=1. if distance>0 else -1.
                    def objective(logradius):
                        c=p0[i]+sign*np.exp(float(logradius[0]))*n
                        return (circle_positions(t,p0[i],v0[i],c)-y).ravel()
                    try:
                        fit=least_squares(objective,[np.log(abs(distance))],bounds=([np.log(1e-5)],[np.log(extent*1e5)]),max_nfev=24)
                        centers.append((p0[i]+sign*np.exp(float(fit.x[0]))*n,1))
                    except (ValueError,ArithmeticError):pass
            for c,fitted in centers:
                try:choices.append({'mode':'circle','center':np.asarray(c).tolist(),'rmse':residual(circle_positions(t,p0[i],v0[i],c)),'fitted_geometry':fitted})
                except (ValueError,ArithmeticError):pass
        # Evidence-based hypothesis selection; observation noise is not ground truth.
        limit=max(2e-6*extent,4*np.sqrt(2)*sigma)
        supported=[c for c in choices if c['rmse']<=limit]
        if not supported:raise BackgroundModelError('No supported autonomous linear/circular background hypothesis for '+ident+'; this is a bridge-model limitation, not a rejection of the probe-force expression')
        selected=min(supported,key=lambda c:c['rmse']**2+c['fitted_geometry']*2*sigma*sigma/max(len(t),1))
        if selected['mode']=='circle':
            cx,cy=selected['center'];props[ident]={'bridge_motion_mode':2.,'bridge_center_x':float(cx),'bridge_center_y':float(cy)}
        audit.append({'body_id':ident,**selected,'observation_count':len(t),'acquired_time_span':[float(t.min()),float(t.max())],'position_sigma':sigma})
    return props,audit

HELPER="def background_acceleration(b):\n    p = b['properties']\n    if p.get('bridge_motion_mode', 0) != 2:\n        return [0.0, 0.0]\n    dx = b['pos'][0] - p['bridge_center_x']\n    dy = b['pos'][1] - p['bridge_center_y']\n    radius2 = dx*dx + dy*dy\n    if radius2 <= 0:\n        raise ValueError('Degenerate autonomous background radius')\n    speed2 = b['vel'][0]*b['vel'][0] + b['vel'][1]*b['vel'][1]\n    return [-speed2*dx/radius2, -speed2*dy/radius2]\n"
