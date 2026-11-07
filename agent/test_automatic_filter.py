"""Automatic SR adapter tests, without PySR, model calls, or official worlds."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import test_trajectory_filter as base
from .pipeline_bridge import automatic_filter
from .pipeline_bridge.filter_tool import install
from .pipeline_bridge.trajectory_filter import consume, PAYLOAD_KEY
from .repository.shore_agent import symbolic_quality, symbolic_regression, symbolic_service


class AutomaticFilterTests(unittest.TestCase):
    def runtime(self, root, probes, enabled=True, policy='automatic'):
        runtime = base.ProtocolTests().runtime(root, probes[0])
        for probe in probes[1:]:runtime.store.add('probe', probe, probe['probe_id'])
        schemas = runtime.schemas
        install(runtime, enabled=enabled, policy=policy)
        if policy == 'automatic':self.assertIs(runtime.schemas, schemas)
        return runtime

    def backend(self, captured):
        def search(probes, budget, **kwargs):
            captured.append((copy.deepcopy(probes), budget))
            original = symbolic_quality.derivatives
            def derivatives(probe, t, r, v):return consume(probe, t, r, v, original)
            with patch.object(symbolic_quality, 'derivatives', derivatives):
                data = symbolic_regression.prepare_dataset(probes)
            for i, count in enumerate(data['samples_per_probe'].values()):
                offset = sum(list(data['samples_per_probe'].values())[:i])
                self.assertAlmostEqual(float(data['weights'][offset:offset+count].sum()), 1/len(probes))
            return {'status': 'completed', 'candidates': [], 'data_quality': {'local_derivative_quality': data['local_quality']}}
        return search

    def test_original_schema_call_automatically_consumed(self):
        p, _ = base.fixture()
        with tempfile.TemporaryDirectory() as root:
            rt = self.runtime(root, [p])
            before = copy.deepcopy(rt.store.records['probe'])
            methods = [s['function']['name'] for s in rt.schemas]
            self.assertNotIn('denoise_trajectories', methods)
            sr = next(s['function'] for s in rt.schemas if s['function']['name'] == 'symbolic_regression')
            self.assertNotIn('preprocessing_id', sr['parameters']['properties'])
            captured = []
            with patch.object(symbolic_service, 'search_remote', self.backend(captured)):
                reply = rt.execute('symbolic_regression', {'train_probe_ids': [p['probe_id']], 'time_budget_seconds': 10.})
            self.assertTrue(reply['ok'], reply)
            data = reply['data'];self.assertTrue(data['preprocessing']['worker_consumption_verified'])
            self.assertEqual(data['automatic_preprocessing']['filtered_count'], 1)
            self.assertEqual(data['automatic_preprocessing']['worker_consumption'], 'verified')
            self.assertEqual(rt.store.records['probe'], before)
            self.assertIn(PAYLOAD_KEY, captured[0][0][0])
            events = [json.loads(line) for line in (Path(root)/'tools.jsonl').read_text().splitlines()]
            self.assertEqual([event['tool'] for event in events], ['symbolic_regression'])
            record = json.loads((Path(root)/'filter-automatic.jsonl').read_text())
            self.assertEqual(record['preprocessing_id'], data['preprocessing']['preprocessing_id'])
            self.assertLessEqual(captured[0][1]+record['preprocessing_seconds'], 10.+1e-9)

    def test_explicit_id_wins_and_bad_id_is_not_rescued(self):
        p, _ = base.fixture()
        with tempfile.TemporaryDirectory() as root:
            rt = self.runtime(root, [p])
            key = rt.denoise_trajectories([p['probe_id']])['preprocessing_id']
            with patch.object(rt, 'denoise_trajectories', side_effect=AssertionError('Do not refilter')):
                with patch.object(symbolic_service, 'search_remote', self.backend([])):
                    reply = rt.symbolic_regression([p['probe_id']], 10., key)
                self.assertTrue(reply['preprocessing']['worker_consumption_verified'])
                with self.assertRaisesRegex(ValueError, 'cross-session'):
                    rt.symbolic_regression([p['probe_id']], 10., 'bad')

    def test_all_bypass_and_zero_noise_are_legacy_not_empty_verified(self):
        for kwargs in [{'n': 8}, {'sigma_r': 0., 'sigma_v': 0.}]:
            p, _ = base.fixture(**kwargs)
            with tempfile.TemporaryDirectory() as root:
                rt = self.runtime(root, [p]);captured = []
                with patch.object(rt, 'denoise_trajectories', side_effect=AssertionError('No eligible probes')):
                    with patch.object(symbolic_service, 'search_remote', self.backend(captured)):
                        result = rt.symbolic_regression([p['probe_id']], 10.)
                audit = result['automatic_preprocessing']
                self.assertEqual(audit['filtered_count'], 0)
                self.assertEqual(audit['worker_consumption'], 'not_required')
                self.assertEqual(audit['filter_calls'], 0)
                self.assertNotIn('preprocessing', result)
                self.assertNotIn(PAYLOAD_KEY, captured[0][0][0])

    def test_mixed_trajectories_keep_separate_weights(self):
        p, _ = base.fixture();q, _ = base.fixture(n=8);q['probe_id'] = 'short'
        with tempfile.TemporaryDirectory() as root:
            rt = self.runtime(root, [p, q]);captured = []
            with patch.object(symbolic_service, 'search_remote', self.backend(captured)):
                result = rt.symbolic_regression([p['probe_id'], q['probe_id']], 10.)
            audit = result['automatic_preprocessing']
            self.assertEqual((audit['filtered_count'], audit['bypass_count']), (1, 1))
            self.assertEqual(audit['worker_consumption'], 'verified')

    def test_numerical_failure_falls_back_but_invalid_raw_is_not_swallowed(self):
        p, _ = base.fixture()
        with tempfile.TemporaryDirectory() as root:
            rt = self.runtime(root, [p]);captured = []
            with patch('agent.pipeline_bridge.filter_tool.filter_probe', side_effect=ValueError('numerical failure')):
                with patch.object(symbolic_service, 'search_remote', self.backend(captured)):
                    result = rt.symbolic_regression([p['probe_id']], 10.)
            self.assertEqual(result['automatic_preprocessing']['filter_status'], 'all_bypass')
            self.assertEqual(result['automatic_preprocessing']['worker_consumption'], 'not_required')
            self.assertNotIn('preprocessing', result)
            self.assertNotIn(PAYLOAD_KEY, captured[0][0][0])
        p['times'][2] = p['times'][1]
        with tempfile.TemporaryDirectory() as root:
            rt = self.runtime(root, [p])
            with patch.object(symbolic_service, 'search_remote', side_effect=AssertionError('Invalid input sent')):
                with self.assertRaisesRegex(ValueError, 'timestamps'):rt.symbolic_regression([p['probe_id']], 10.)

    def test_disabled_and_explicit_without_id_stay_raw(self):
        p, _ = base.fixture()
        for enabled, policy in [(False, 'automatic'), (True, 'explicit')]:
            with tempfile.TemporaryDirectory() as root:
                rt = self.runtime(root, [p], enabled=enabled, policy=policy);captured = []
                with patch.object(symbolic_service, 'search_remote', self.backend(captured)):
                    result = rt.symbolic_regression([p['probe_id']], 10.)
                self.assertNotIn('automatic_preprocessing', result)
                self.assertNotIn(PAYLOAD_KEY, captured[0][0][0])
                self.assertEqual(captured[0][1], 10.)
                self.assertFalse((Path(root)/'filter-automatic.jsonl').exists())

    def test_shared_budget_debits_artifact_cost_and_reserves_finalization(self):
        p, _ = base.fixture();clock = [100.];budgets = []
        with tempfile.TemporaryDirectory() as root:
            runtime = SimpleNamespace(run_dir=root, task={'position_noise_sigma': .04, 'velocity_noise_sigma': .0003},
                                      _probes=lambda ids: [copy.deepcopy(p)],
                                      budget=lambda: {'wall_seconds_remaining': 200.-(clock[0]-100.)})
            def denoise(ids, time_budget_seconds):
                self.assertEqual(time_budget_seconds, 2.)
                clock[0] += 2.4  # Include serialization/overrun, not declared cap.
                return {'preprocessing_id': 'artifact', 'per_probe': [{'probe_id': p['probe_id'], 'status': 'filtered'}]}
            runtime.denoise_trajectories = denoise
            def reference(ids, budget, key):
                budgets.append(budget);clock[0] += 1
                return {'preprocessing': {'worker_consumption_verified': True, 'preprocessing_id': key}}
            auto = automatic_filter.wrap(runtime, lambda *args: self.fail('Unexpected legacy'), reference)
            with patch.object(automatic_filter.time, 'monotonic', lambda: clock[0]):
                result = auto([p['probe_id']], 10.)
            self.assertAlmostEqual(budgets[0], 7.6)
            self.assertAlmostEqual(result['automatic_preprocessing']['total_seconds'], 3.4)
            runtime.budget = lambda: {'wall_seconds_remaining': 36.5}
            auto = automatic_filter.wrap(runtime, lambda ids, budget: {'budget': budget}, lambda *args: self.fail('No time for filtering'))
            runtime.denoise_trajectories = lambda *args, **kwargs: self.fail('No time for filtering')
            result = auto([p['probe_id']], 10.)
            self.assertLessEqual(result['budget'], 1.5)
            self.assertEqual(result['automatic_preprocessing']['filter_status'], 'insufficient_budget')

    def test_invalid_budgets_rejected_without_filter(self):
        p, _ = base.fixture()
        with tempfile.TemporaryDirectory() as root:
            rt = self.runtime(root, [p])
            for budget in [True, 0., -1., float('nan'), float('inf'), '10']:
                with self.subTest(budget=budget):
                    with self.assertRaises(ValueError):rt.symbolic_regression([p['probe_id']], budget)

    def test_worker_consumption_failure_is_audited_as_unverified(self):
        p, _ = base.fixture()
        with tempfile.TemporaryDirectory() as root:
            rt = self.runtime(root, [p])
            with patch.object(symbolic_service, 'search_remote', return_value={'status': 'completed', 'candidates': []}):
                with self.assertRaisesRegex(ValueError, 'did not acknowledge'):
                    rt.symbolic_regression([p['probe_id']], 10.)
            audit = json.loads((Path(root)/'filter-automatic.jsonl').read_text())
            self.assertEqual(audit['filtered_count'], 1)
            self.assertEqual(audit['worker_consumption'], 'unverified')
            self.assertEqual(audit['status'], 'failed')

    def test_preprocessing_overrun_never_starts_budgetless_search(self):
        p, _ = base.fixture();clock = [100.]
        with tempfile.TemporaryDirectory() as root:
            runtime = SimpleNamespace(run_dir=root, task={'position_noise_sigma': .04, 'velocity_noise_sigma': .0003},
                                      _probes=lambda ids: [copy.deepcopy(p)], budget=lambda: {'wall_seconds_remaining': 200.})
            def denoise(*args, **kwargs):
                clock[0] += 10.
                return {'preprocessing_id': 'artifact', 'per_probe': [{'probe_id': p['probe_id'], 'status': 'filtered'}]}
            runtime.denoise_trajectories = denoise
            def forbidden(*args):self.fail('No budget remains for SR')
            auto = automatic_filter.wrap(runtime, forbidden, forbidden)
            with patch.object(automatic_filter.time, 'monotonic', lambda: clock[0]):
                with self.assertRaisesRegex(ValueError, 'exhausted'):auto([p['probe_id']], 10.)
            audit = json.loads((Path(root)/'filter-automatic.jsonl').read_text())
            self.assertEqual(audit['search_seconds'], 0.)
            self.assertEqual(audit['status'], 'failed')


if __name__ == '__main__':unittest.main()
