"""Only public Console observations; local protocol/shape compatibility."""
import copy
import time
import uuid
import numpy as np
from rift.console import ConsoleError
from .compiler import property_mapping


class Environment:
    ruleset = 'e4'
    def __init__(self, console):
        self.console = console
        self.public_mission = None
        self.deadline = None
        self.last_budget = {}
        self.ended = False
        self.serial = 0
        self.acquired = {}

    def update(self, response, request_started=None):
        budget = response.get('budget', {})
        self.last_budget.update(budget)
        if 'wall_clock_remaining_seconds' in budget:
            # Anchor at request start: network time cannot extend the deadline.
            deadline = (request_started or time.monotonic()) + budget['wall_clock_remaining_seconds']
            self.deadline = min(self.deadline, deadline) if self.deadline is not None else deadline
        return response

    def budget(self):
        b = self.last_budget
        remaining = max(0., self.deadline-time.monotonic()) if self.deadline is not None else 300.
        return {'energy_remaining': b.get('energy_remaining'), 'energy_used': b.get('energy_used', 0),
                'observations_remaining': b.get('observe_remaining'),
                'wall_seconds_remaining': remaining, 'session_ended': self.ended or remaining <= 0}

    def call(self, method, **kwargs):
        started = time.monotonic()
        return self.update(getattr(self.console, method)(**kwargs), started)

    def converted_probes(self, record, identifiers=None, spacing=None):
        result = []
        mapping = property_mapping(self.public_mission)
        for index, initial in enumerate(record['probes']):
            if identifiers is None:
                self.serial += 1
                identifier = f'probe_{self.serial:04d}'
            else:
                identifier = identifiers[index]
            exact = copy.deepcopy(record)
            exact['probes'] = [copy.deepcopy(initial)]
            exact['positions'] = np.asarray(record['positions'])[:, index:index+1, :].tolist()
            exact['velocities'] = np.asarray(record['velocities'])[:, index:index+1, :].tolist()
            times = np.asarray(record['times'], dtype=float)
            velocities = np.asarray(record['velocities'])[:, index, :]
            properties = {}
            preferred=[k for k in ('mass','charge') if k in mapping]
            preferred += [k for k in mapping if k not in preferred and k not in ('p1','p2')]
            for alias in preferred:
                scope,key=mapping[alias]
                values = initial['properties'] if scope == 'probe' else record.get('source_properties', {})
                if key in values:
                    properties[alias] = values[key]
            uniform=len(times)<3 or np.allclose(np.diff(times),np.diff(times)[0],rtol=0,atol=1e-10)
            if len(times)<3:
                derivative=np.empty((0,2));method='unavailable_fewer_than_3_samples'
            elif uniform:
                step=spacing if spacing is not None else float(record['readout'][1]-record['readout'][0])
                derivative=(velocities[2:]-velocities[:-2])/(2*step);method='velocity_centered_difference'
            else:
                derivative=np.gradient(velocities,times,axis=0)[1:-1];method='velocity_nonuniform_centered_difference'
            result.append({'probe_id': identifier,
                'initial_conditions': {'r0': initial['pos'], 'v0': initial['vel'],
                    't0': record['t_launch'], 'properties': properties},
                'times': times.tolist(), 'r_obs': np.asarray(record['positions'])[:, index, :].tolist(),
                'v_obs': velocities.tolist(),
                'a_est': derivative.tolist(),'a_est_times': times[1:-1].tolist(),'a_est_method':method, '_official_record': exact})
        self.acquired.update({p['probe_id']: copy.deepcopy(p) for p in result})
        return result

    def mission(self):
        if self.public_mission is None:
            self.public_mission=self.call('mission')
            self.initial_probes=self.converted_probes(self.public_mission['first_light'])
        public=self.public_mission;mapping=property_mapping(public)
        names=[k for k in ('mass','charge') if k in mapping]
        names += [k for k in mapping if k not in names and k not in ('p1','p2')]
        region=public['launch_region'];bg=public['background_initial_state'];limits=public['limits']
        reference=float(np.sqrt((region['radius_min']**2+region['radius_max']**2)/2)) if region.get('kind')=='annulus' else None
        ident=str(public['world_id']);wid='world_'+ident if ident.isdecimal() else ident
        task={'name':'Local physics discovery · v15 public worlds','dimension':2,'world_set':'v15',
            'world_id':wid,'world_name':'世界 '+ident.zfill(2),'physics_version':'shore_v15',
            'numerics_version':public['numerics_version'],'ruleset':self.ruleset,
            'units':{'position':'local_length','time':'local_time','velocity':'local_length/local_time'},
            'body_ids':bg['ids'],'body_observation_protocol':'time_series',
            'body_initial_states':[{'body_id':bid,'position':pos,'velocity':vel} for bid,pos,vel in zip(bg['ids'],bg['positions'],bg['velocities'])],
            'body_position_source':'body_initial_states gives exact states only at absolute t=0. At later times use acquired body observations.',
            'allowed_properties':names,
            'property_controls':{k:copy.deepcopy(public['controls'][scope+'_properties'][key]) for k in names for scope,key in [mapping[k]]},
            'property_mapping':{k:'public '+scope+' control '+key+(' for this probe experiment' if scope=='source' else '') for k in names for scope,key in [mapping[k]]},
            'property_note':'Each probe in a batch is an independent experiment. Its mass maps to p2 and charge maps to source p1; omitted values default to 1. The names are compatibility aliases, not claims about their physical effects.',
            'reference_length':reference,'position_noise_sigma':public['noise'].get('position_sigma',.01*reference if reference is not None else None),
            'velocity_noise_sigma':public['noise']['velocity_sigma'],
            'noise_note':'Independent Gaussian position and probe-velocity noise on each new measurement when noise is enabled; cached records are immutable. Controlled initial states are exact.',
            'local_config_notice':'v15 public physical domains with '+self.ruleset+' local rules: 10 energy, 300 seconds. This remains a compatibility adapter, not the standard contest Console.',
            'position_limit':region.get('radius_max'),'velocity_limit':public['initial_speed_max'],
            'launch_region':copy.deepcopy(region),'initial_speed_max':public['initial_speed_max'],
            'maximum_single_trajectory_duration':public['maximum_single_trajectory_duration'],
            'physical_time_range':[0,public['maximum_physical_time']],'time_grid':.005,
            'limits':{'initial_energy':limits['energy_budget'],'energy_budget':limits['energy_budget'],
                'probe_energy':limits['energy_per_new_probe'],'max_batch_probes':limits['max_probes_per_launch'],
                'max_probe_samples':limits['max_readouts_per_probe'],'max_observe_calls':limits['observe_calls'],
                'max_observe_times':limits['max_times_per_observe'],'wall_seconds':limits['wall_clock_seconds']},
            'sampling_note':'Absolute t0 plus uniform dt for insert_probe; n_samples includes t0. Thinking does not advance physical time. The supplied first-light times are nonuniform.',
            'first_light_note':'Two free eight-sample public first-light probes plus background positions at matching times. Repeated mission reads return cached data and do not reset the session.'}
        if 'charge' not in names:
            task['property_note']='mass is the public probe control; an omitted value defaults to 1.'
        light=public['first_light']
        body={'observation_id':'body_first_light','times':light['times'],'body_ids':light['background_ids'],'positions_obs':light['visible_background_positions']}
        return {'task':task,'first_light':{'probes':[{k:copy.deepcopy(v) for k,v in p.items() if k!='_official_record'} for p in self.initial_probes], 'body_observation':body}}

    def observe_body(self, times):
        raw = self.call('observe', times=times)
        return {'observation_id': 'body_' + uuid.uuid4().hex[:12], 'times': raw.get('times', times),
                'body_ids': raw.get('background_ids', raw.get('ids')),
                'positions_obs': raw.get('positions', raw.get('visible_background_positions'))}

    def insert_probe(self, probes, t0, dt, n_samples):
        if not isinstance(dt, (int, float)) or dt <= 0:
            raise ValueError('dt must be positive')
        mapping = property_mapping(self.public_mission)
        controls = self.public_mission['controls']
        groups = {}
        for index, probe in enumerate(probes):
            pp = {k:v['default'] for k,v in controls['probe_properties'].items()}
            sp = {k:v['default'] for k,v in controls['source_properties'].items()}
            seen = {}
            for alias,value in probe.get('properties', {}).items():
                if alias not in mapping:
                    raise ValueError('Property unavailable in this mission: ' + alias)
                scope,key = mapping[alias]
                if (scope,key) in seen and seen[scope,key] != value:
                    raise ValueError('Conflicting public property aliases')
                seen[scope,key] = value
                (pp if scope=='probe' else sp)[key] = value
            group = groups.setdefault(tuple(sorted(sp.items())), [])
            group.append((index, {'pos': probe['r0'], 'vel': probe['v0'], 'properties': pp}))
        reserved={i:f'probe_{self.serial+i+1:04d}' for i in range(len(probes))}
        self.serial += len(probes)
        acquired = []
        for source, group in groups.items():
            raw = self.call('launch', t_launch=t0, readout=[i*dt for i in range(n_samples)],
                probes=[p for _,p in group], source_properties=dict(source))
            converted = self.converted_probes(raw,[reserved[index] for index,_ in group],spacing=dt)
            acquired.extend((index,p) for (index,_),p in zip(group, converted))
        return {'probes': [{k:copy.deepcopy(v) for k,v in p.items() if k!='_official_record'} for _,p in sorted(acquired)]}

    def end_session(self):
        reply=self.backend.commit_submission()
        self.ended=True
        return {'session_ended':True,'official_reply':reply}

    def close(self):
        pass

    def request(self, name, **kwargs):
        try:
            data = None if name == 'budget' else getattr(self, name)(**kwargs)
            return {'ok': True, 'data': data, 'error': None, 'budget': self.budget()}
        except ConsoleError as exc:
            return {'ok': False, 'data': None, 'error': exc.as_dict(), 'budget': self.budget()}
