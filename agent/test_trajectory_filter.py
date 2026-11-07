"""Lightweight observation-only numerical/protocol tests; no rift/PySR/model."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from .pipeline_bridge.trajectory_filter import filter_probe, consume, raw_digest, digest, VERSION, PAYLOAD_KEY, _basis, capability, annotate_report
from .pipeline_bridge.filter_tool import install, with_noise
from .pipeline_bridge.bootstrap import repository
repository()
from .repository.shore_agent.runtime import ToolRuntime
from .repository.shore_agent import symbolic_quality, symbolic_regression, symbolic_service


def fixture(seed=1, n=40, frequency=1., sigma_r=.04, sigma_v=.0003, nonuniform=False):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8, n)
    if nonuniform:
        t = 8*(t/8)**1.3
    f = frequency
    r = np.c_[np.cos(f*t), np.sin(.7*f*t)]
    v = np.c_[-f*np.sin(f*t), .7*f*np.cos(.7*f*t)]
    a = np.c_[-f*f*np.cos(f*t), -(.7*f)**2*np.sin(.7*f*t)]
    p = {'probe_id': 'probe_test', 'times': t.tolist(),
         'r_obs': (r+rng.normal(0, sigma_r, r.shape)).tolist(),
         'v_obs': (v+rng.normal(0, sigma_v, v.shape)).tolist(),
         'initial_conditions': {'t0': 0., 'r0': r[0].tolist(), 'v0': v[0].tolist(), 'properties': {'mass': 1.}},
         'observation_noise': {'position_sigma': sigma_r, 'velocity_sigma': sigma_v}}
    return p, (r, v, a)


class NumericalTests(unittest.TestCase):
    def test_input_and_noisy_aggregate(self):
        rows = []
        legacy = []
        for seed in range(5):
            p, truth = fixture(seed)
            before = copy.deepcopy(p)
            out = filter_probe(p, deadline=time.monotonic()+2)
            self.assertEqual(p, before)
            self.assertEqual(out['status'], 'filtered')
            fitted = [np.asarray(out['arrays'][k]) for k in 'rva']
            # Compare exactly the same interior timestamps consumed by SR.
            rows.append([float(np.mean((x[1:-1]-y[1:-1])**2)) for x, y in zip(fitted, truth)])
            t, r, v = [np.asarray(p[k]) for k in ('times', 'r_obs', 'v_obs')]
            old = symbolic_quality.derivatives(p, t, r, v)
            legacy.append([float(np.mean((x-y[1:-1])**2)) for x, y in zip(old[:3], truth)])
        # Benefit and protocol correctness are separate: acceleration can worsen
        # versus legacy for this family/seed. Do not tune it to make a test pass.
        self.assertTrue(np.all(np.mean(rows, axis=0)[:2] < np.mean(legacy, axis=0)[:2]))
        self.assertLess(np.sqrt(np.mean(rows, axis=0)[2]), .01)

    def test_units_rotation_and_absolute_time(self):
        p, _ = fixture(nonuniform=True)
        out = filter_probe(p)
        transformed = copy.deepcopy(p)
        c, b, shift = 7., .3, 123.
        rotation = np.array([[.6, -.8], [.8, .6]])
        transformed['times'] = (np.array(p['times'])*b+shift).tolist()
        transformed['r_obs'] = (np.array(p['r_obs']) @ rotation.T*c).tolist()
        transformed['v_obs'] = (np.array(p['v_obs']) @ rotation.T*c/b).tolist()
        transformed['initial_conditions'].update(t0=shift, r0=(np.array(p['initial_conditions']['r0']) @ rotation.T*c).tolist(), v0=(np.array(p['initial_conditions']['v0']) @ rotation.T*c/b).tolist())
        transformed['observation_noise'] = {'position_sigma': .04*c, 'velocity_sigma': .0003*c/b}
        other = filter_probe(transformed)
        self.assertEqual(out['lambda'], other['lambda'])
        for order, key in enumerate('rva'):
            np.testing.assert_allclose(np.array(other['arrays'][key]), np.array(out['arrays'][key]) @ rotation.T*c/b**order, rtol=2e-5, atol=2e-6)

    def test_single_zero_channels(self):
        for sr, sv in [(0., .0003), (.04, 0.)]:
            p, _ = fixture(sigma_r=sr, sigma_v=sv, nonuniform=True)
            out = filter_probe(p)
            key = 'r' if sr == 0 else 'v'
            np.testing.assert_allclose(out['arrays'][key], p[key+'_obs'], atol=2e-7)

    def test_bypass_and_time_budget(self):
        for n in [3, 5, 8, 11]:
            p, _ = fixture(n=n)
            self.assertEqual(filter_probe(p)['reason'], 'short_trajectory')
        p, _ = fixture(sigma_r=0, sigma_v=0)
        self.assertEqual(filter_probe(p)['reason'], 'zero_noise')
        with self.assertRaises(TimeoutError):
            filter_probe(fixture()[0], deadline=0)

    def test_invalid_inputs(self):
        for change in ['time', 'noise', 'nan']:
            p, _ = fixture()
            if change == 'time':p['times'][2] = p['times'][1]
            if change == 'noise':p['observation_noise']['position_sigma'] = -1
            if change == 'nan':p['v_obs'][2][0] = float('nan')
            with self.assertRaises(ValueError):filter_probe(p)

    def test_fast_dynamics_preserved_when_resolved(self):
        p, truth = fixture(n=96, frequency=3.)
        out = filter_probe(p)
        af = np.array(out['arrays']['a'])[2:-2]
        a = truth[2][2:-2]
        self.assertLess(np.sqrt(np.mean((af-a)**2))/np.sqrt(np.mean(a*a)), .1)
        self.assertGreater(np.linalg.norm(af)/np.linalg.norm(a), .9)

    def test_constant_acceleration(self):
        p, _ = fixture()
        t = np.array(p['times'])[:, None]
        a = np.array([.17, -.08]);v0 = np.array([.4, .7]);r0 = np.array([1., -2.])
        rng = np.random.default_rng(31)
        p['r_obs'] = (r0+v0*t+.5*a*t*t+rng.normal(0, .04, (len(t), 2))).tolist()
        p['v_obs'] = (v0+a*t+rng.normal(0, .0003, (len(t), 2))).tolist()
        p['initial_conditions'].update(r0=r0.tolist(), v0=v0.tolist())
        out = filter_probe(p)
        np.testing.assert_allclose(out['arrays']['a'], np.tile(a, (len(t), 1)), atol=.005)

    def test_late_first_readout_and_kinematics(self):
        p, truth = fixture(n=160)
        p['times'], p['r_obs'], p['v_obs'] = [p[k][3:] for k in ('times', 'r_obs', 'v_obs')]
        out = filter_probe(p)
        t = np.array(p['times'])
        r, v, a = [np.array(out['arrays'][k]) for k in 'rva']
        np.testing.assert_allclose(np.gradient(r, t, axis=0)[2:-2], v[2:-2], atol=.001)
        # Coarse sampled finite differences have truncation error and cannot
        # establish exact consistency of a continuously differentiable spline.
        self.assertLess(np.sqrt(np.mean((np.gradient(v, t, axis=0)[2:-2]-a[2:-2])**2)), .01)

    def test_continuous_basis_derivatives(self):
        knots = np.r_[np.zeros(6), [.13, .39, .42, .8], np.ones(6)]
        coef = np.random.default_rng(19).normal(size=(len(knots)-6, 2))
        points = np.array([.05, .2, .41, .65, .9])
        step = 1e-6
        for derivative in [0, 1]:
            difference = ((_basis(knots, points+step, derivative)-_basis(knots, points-step, derivative)) @ coef)/(2*step)
            np.testing.assert_allclose(difference, _basis(knots, points, derivative+1) @ coef, rtol=2e-7, atol=2e-7)


class ProtocolTests(unittest.TestCase):
    def runtime(self, root, p):
        class Environment:
            ruleset = 'e5'
            def request(self, name, **kwargs):
                return {'ok': True, 'budget': {'wall_seconds_remaining': 200., 'session_ended': False}}
        rt = ToolRuntime(Environment(), root, config={'symbolic_regression_enabled': True})
        rt.task = {'position_noise_sigma': p['observation_noise']['position_sigma'],
                   'velocity_noise_sigma': p['observation_noise']['velocity_sigma']}
        rt.store.add('probe', p, p['probe_id'])
        return rt

    def test_disabled_is_noop_and_source_integrity(self):
        p, _ = fixture()
        with tempfile.TemporaryDirectory() as root:
            rt = self.runtime(root, p)
            original = rt.schemas
            state = dict(rt.__dict__)
            install(rt, enabled=False)
            self.assertIs(original, rt.schemas)
            self.assertEqual(set(rt.__dict__), set(state))
            self.assertFalse(hasattr(rt, 'denoise_trajectories'))
        manifest = json.loads((Path(__file__).parent/'repository-source.json').read_text())
        for name, value in manifest['files'].items():
            self.assertEqual(hashlib.sha256((Path(__file__).parent/'repository/shore_agent'/name).read_bytes()).hexdigest(), value)

    def test_runtime_to_transport_to_dataset_raw_kept(self):
        p, _ = fixture()
        with tempfile.TemporaryDirectory() as root:
            rt = self.runtime(root, p)
            install(rt, enabled=True)
            response = rt.execute('denoise_trajectories', {'train_probe_ids': [p['probe_id']]})
            self.assertTrue(response['ok'], response)
            key = response['data']['preprocessing_id']
            raw_before = copy.deepcopy(rt.store.records['probe'])
            def search(probes, budget, **kwargs):
                payload, _ = symbolic_service._request(probes, budget, 42)
                # Real package dataset builder, opted-in worker derivative adapter.
                original = symbolic_quality.derivatives
                def derivatives(probe, t, r, v):return consume(probe, t, r, v, original)
                with patch.object(symbolic_quality, 'derivatives', derivatives):
                    dataset = symbolic_regression.prepare_dataset(json.loads(json.dumps(payload))['probes'])
                np.testing.assert_allclose(dataset['weights'].sum(), 1.)
                self.assertEqual(dataset['probes'][0]['r_obs'], p['r_obs'])
                self.assertEqual(dataset['probes'][0]['v_obs'], p['v_obs'])
                np.testing.assert_allclose(dataset['y'], probes[0][PAYLOAD_KEY]['arrays']['a'][1:-1])
                return {'status': 'completed', 'candidates': [], 'data_quality': {'local_derivative_quality': dataset['local_quality']}}
            with patch.object(symbolic_service, 'search_remote', search):
                response = rt.execute('symbolic_regression', {'train_probe_ids': [p['probe_id']], 'time_budget_seconds': 10., 'preprocessing_id': key})
            self.assertTrue(response['ok'], response)
            self.assertTrue(response['data']['preprocessing']['worker_consumption_verified'])
            self.assertEqual(rt.store.records['probe'], raw_before)
            self.assertIn('analyses/'+key+'.json', rt.files.entries())
            # Same tool ID in another session is rejected.
            with tempfile.TemporaryDirectory() as other:
                second = self.runtime(other, p);install(second, enabled=True)
                with self.assertRaisesRegex(ValueError, 'cross-session'):
                    second.symbolic_regression([p['probe_id']], 10, key)
            rt.task['position_noise_sigma'] *= 2
            with self.assertRaisesRegex(ValueError, 'Stale'):
                rt.symbolic_regression([p['probe_id']], 10, key)

    def test_subprocess_consumer_and_hash_rejection(self):
        p, _ = fixture()
        p[PAYLOAD_KEY] = filter_probe(p)
        script = """import json,sys
from agent.pipeline_bridge.bootstrap import repository
repository()
from agent.pipeline_bridge.trajectory_filter import install_consumer
install_consumer()
from agent.repository.shore_agent.symbolic_regression import prepare_dataset
d=prepare_dataset(json.load(sys.stdin));print(json.dumps({'y':d['y'].tolist(),'quality':d['local_quality']}))
"""
        result = subprocess.run([sys.executable, '-B', '-c', script], input=json.dumps([p]), text=True, capture_output=True, check=True)
        d = json.loads(result.stdout)
        np.testing.assert_allclose(d['y'], p[PAYLOAD_KEY]['arrays']['a'][1:-1])
        self.assertEqual(d['quality'][0]['method'], VERSION)
        p[PAYLOAD_KEY]['arrays']['a'][2][0] += 1
        t, r, v = [np.array(p[k]) for k in ('times', 'r_obs', 'v_obs')]
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):consume(p, t, r, v, symbolic_quality.derivatives)

    def test_zero_noise_and_short_legacy_exact(self):
        for n, sr, sv in [(8, .04, .0003), (40, 0., 0.)]:
            p, _ = fixture(n=n, sigma_r=sr, sigma_v=sv)
            t, r, v = [np.array(p[k]) for k in ('times', 'r_obs', 'v_obs')]
            expected = symbolic_quality.derivatives(p, t, r, v)
            p[PAYLOAD_KEY] = filter_probe(p)
            actual = consume(p, t, r, v, symbolic_quality.derivatives)
            for a, b in zip(expected[:5], actual[:5]):np.testing.assert_array_equal(a, b)

    def test_old_worker_is_rejected_and_legacy_call_stays_raw(self):
        p, _ = fixture()
        with tempfile.TemporaryDirectory() as root:
            rt = self.runtime(root, p);install(rt, enabled=True)
            key = rt.denoise_trajectories([p['probe_id']])['preprocessing_id']
            payloads = []
            def old_worker(probes, budget, **kwargs):
                payloads.append(copy.deepcopy(probes))
                return {'status': 'completed', 'candidates': []}
            original = rt._probes
            with patch.object(symbolic_service, 'search_remote', old_worker):
                with self.assertRaisesRegex(ValueError, 'did not acknowledge'):
                    rt.symbolic_regression([p['probe_id']], 10., key)
                self.assertEqual(rt._probes, original)
                rt.symbolic_regression([p['probe_id']], 10.)
            self.assertIn(PAYLOAD_KEY, payloads[0][0])
            self.assertNotIn(PAYLOAD_KEY, payloads[1][0])
            self.assertEqual(payloads[1], with_noise(rt, [p]))

    def test_tiny_budget_fallback_and_bad_reference(self):
        p, _ = fixture()
        with tempfile.TemporaryDirectory() as root:
            rt = self.runtime(root, p);install(rt, enabled=True)
            reply = rt.denoise_trajectories([p['probe_id']], time_budget_seconds=1e-12)
            self.assertEqual(reply['per_probe'][0]['reason'], 'budget_exhausted')
            with self.assertRaisesRegex(ValueError, 'selection'):
                rt.symbolic_regression([], 10., reply['preprocessing_id'])

    def test_resolution_retained_without_resmoothing_joint(self):
        p, _ = fixture();p[PAYLOAD_KEY] = filter_probe(p)
        t, r, v = [np.array(p[k]) for k in ('times', 'r_obs', 'v_obs')]
        expected = symbolic_quality.derivatives(p, t, r, v)
        def raw_only(probe, ti, ri, vi):
            np.testing.assert_array_equal(ri, p['r_obs'])
            np.testing.assert_array_equal(vi, p['v_obs'])
            return symbolic_quality.derivatives(probe, ti, ri, vi)
        actual = consume(p, t, r, v, raw_only)
        np.testing.assert_array_equal(actual[3], expected[3])
        np.testing.assert_array_equal(actual[4], expected[4])
        self.assertGreater(float(np.max(actual[4])), 0.)
        report = annotate_report({'warnings': ['Search loss uses finite-difference acceleration, not independent prediction.']}, [p])
        self.assertEqual(report['target'], 'joint_spline_acceleration')
        self.assertNotIn('Search loss uses finite-difference acceleration, not independent prediction.', report['warnings'])

    def test_capability_handshake_bounded_loopback(self):
        from .pipeline_bridge.filter_capability import verify
        from .pipeline_bridge.service import symbolic_service as external_service
        state = {'trajectory_filter': capability(), 'pool_size': 4}
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200);self.end_headers();self.wfile.write(json.dumps(state).encode())
            def log_message(self, *args):pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
        endpoint = 'http://127.0.0.1:'+str(server.server_port)
        try:
            self.assertEqual(verify(endpoint)['pool_size'], 4)
            with tempfile.TemporaryDirectory() as root:
                with external_service(root, external_endpoint=endpoint, trajectory_filter=True) as actual:
                    self.assertEqual(actual, endpoint)
                self.assertTrue((Path(root)/'filter-worker-capability.json').exists())
            state['trajectory_filter'] = {'version': VERSION, 'consumer_sha256': 'old'}
            with self.assertRaisesRegex(ValueError, 'mismatch'):verify(endpoint)
            state.clear()
            with self.assertRaisesRegex(ValueError, 'missing'):verify(endpoint)
            state.update(trajectory_filter_status={'expected': capability(), 'pending_workers': 1, 'mismatched_workers': 0})
            timer = threading.Timer(.05, lambda: state.update(trajectory_filter=capability(), pool_size=4))
            timer.start()
            self.assertEqual(verify(endpoint, wait_timeout=1.)['pool_size'], 4)
            timer.join()
            state.pop('trajectory_filter')
            with self.assertRaisesRegex(ValueError, 'wait exhausted'):verify(endpoint, wait_timeout=.01)
            for bad in ['https://127.0.0.1:80', 'http://example.com:80', endpoint+'?x=1', 'http://user@127.0.0.1:80']:
                with self.assertRaises(ValueError):verify(bad)
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_advertise_busy_and_restarting_workers(self):
        from .pipeline_bridge.filter_capability import advertise
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'ready.json'
            path.write_text(json.dumps({'trajectory_filter': capability()}))
            worker = symbolic_service._Worker(SimpleNamespace(poll=lambda: None), Path(root), None)
            worker.busy = True
            self.assertTrue(worker.ready())  # ready() means alive/warmed, not idle.
            pool = SimpleNamespace(workers=[worker], lock=threading.Lock(), health=lambda: {'status': 'busy', 'pool_size': 1})
            advertise(pool)
            self.assertEqual(pool.health()['trajectory_filter'], capability())
            pool.workers.append(None)  # A hard-timeout worker is being rebuilt.
            health = pool.health()
            self.assertNotIn('trajectory_filter', health)
            self.assertEqual(health['trajectory_filter_status']['pending_workers'], 1)
            pool.workers.pop()
            path.write_text(json.dumps({'trajectory_filter': {'version': 'old'}}))
            self.assertEqual(pool.health()['trajectory_filter_status']['mismatched_workers'], 1)


if __name__ == '__main__':unittest.main()
