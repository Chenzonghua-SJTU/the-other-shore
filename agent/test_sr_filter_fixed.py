"""E16 control/evaluation unit tests. No Julia, models, or development worlds."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .compare_sr_filter_fixed import select_candidate, read_inputs, save
from .evaluate_sr_filter_fixed import recovery
from . import evaluate_sr_filter_fixed as evaluator
from . import run_sr_filter_fixed as controller


class FixedSRTests(unittest.TestCase):
    def test_selection_uses_only_training_error_and_rank(self):
        rows = [{'sr_rank': 1, 'raw_position_rmse': 1., 'law_file': 'a.py', 'hidden_score': 100.},
                {'sr_rank': 2, 'raw_position_rmse': .1, 'law_file': 'b.py', 'hidden_score': 0.}]
        self.assertEqual(select_candidate(rows), 2)
        rows[0]['raw_position_rmse'] = .1
        self.assertEqual(select_candidate(rows), 1)
        rows[0]['raw_position_rmse'] = float('nan')
        self.assertEqual(select_candidate(rows), 2)
        self.assertIsNone(select_candidate([]))

    def test_fixed_inputs_and_constants_are_identical(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'R07/world-1/agent-output'
            save(path/'mission.json', {'task': {'position_noise_sigma': .04, 'velocity_noise_sigma': .0003}})
            save(path/'result.json', {'submission': {'claims': {'dynamics_order': 2}}, 'budget': {'energy_used': 7}})
            probe_path = path/'records/probe/p.json';save(probe_path, {'probe_id': 'p', 'r_obs': [[1., 2.]]})
            before = probe_path.read_bytes()
            a, ca = read_inputs(root, 1);b, cb = read_inputs(root, 1)
            self.assertEqual(a, b);self.assertEqual(ca, cb)
            self.assertEqual(ca['energy_used'], 7)
            self.assertEqual(before, probe_path.read_bytes())

    def test_summary_keeps_failures_and_original_rank_separate(self):
        rows = []
        for arm, score, rank1_score in [('legacy', 80., 30.), ('joint', 90., 40.)]:
            rows.append({'world': 1, 'seed': 42, 'arm': arm, 'status': 'completed', 'evaluation': {
                'selected': {'score': {'total': score, 'prediction': score-15.}, 'recovery': {'correct': True}},
                'sr_rank_one': {'score': {'total': rank1_score}, 'recovery': {'correct': False}},
                'any_top_three_recovered': True}})
        rows.append({'world': 1, 'seed': 43, 'arm': 'joint', 'status': 'failed'})
        with tempfile.TemporaryDirectory() as root:
            result = controller.summarize(Path(root), rows)
        self.assertEqual(result['arms']['joint']['attempts'], 2)
        self.assertEqual(result['arms']['joint']['mean_total'], 45.)
        self.assertEqual(result['paired_results'][0]['score_delta'], 10.)
        self.assertEqual(result['arms']['legacy']['original_sr_rank_one_recovered'], 0)

    def test_recovery_uses_reference_only_in_evaluator(self):
        evaluator._REFERENCE_CACHE.clear()
        config = {'launch_region': {'kind': 'annulus', 'center': [0., 0.], 'radius_min': 1., 'radius_max': 2.},
                  'initial_speed_max': 1., 'maximum_physical_time': 4., 'probe_properties': {}, 'source_properties': {}}
        class Reference:
            def __init__(self, world):pass
            def acceleration(self, t, positions, **kwargs):return -positions
        exact = SimpleNamespace(params={}, acceleration=lambda t, bodies, params: np.array([-b['pos'] for b in bodies]))
        wrong = SimpleNamespace(params={}, acceleration=lambda t, bodies, params: np.array([-2*b['pos'] for b in bodies]))
        modules = {'development_worlds': SimpleNamespace(catalog=SimpleNamespace(config=lambda world: config)),
                   'development_worlds.worlds': SimpleNamespace(DevelopmentSimulator=Reference),
                   'rift': SimpleNamespace(load_law=lambda path: exact if path == 'exact' else wrong)}
        public = {'background_initial_state': {'ids': []}, 'source_property_target_id': None}
        with patch.dict(sys.modules, modules):
            self.assertTrue(recovery(1, 'exact', public)['correct'])
            outcome = recovery(1, 'wrong', public)
            self.assertFalse(outcome['correct']);self.assertAlmostEqual(outcome['normalized_acceleration_rmse'], 1.)

    def test_reference_time_background_and_spurious_velocity(self):
        evaluator._REFERENCE_CACHE.clear()
        config = {'launch_region': {'kind': 'annulus', 'center': [0., 0.], 'radius_min': 1., 'radius_max': 2.},
                  'initial_speed_max': 1., 'maximum_physical_time': 4., 'probe_properties': {}, 'source_properties': {}}
        class Reference:
            def __init__(self, world):pass
            def acceleration(self, t, positions, *, source_properties=None, probe_properties=None):return -positions
            def background(self, times):
                return {'times': times, 'ids': ['B'],
                        'positions': np.array([[[t, 0.]] for t in times]),
                        'velocities': np.array([[[1., 0.]] for t in times])}
        def candidate(t, bodies, params):
            np.testing.assert_allclose(bodies[0]['pos'], [t, 0.])
            np.testing.assert_allclose(bodies[0]['vel'], [1., 0.])
            return np.array([[0., 0.], -bodies[-1]['pos']+bodies[-1]['vel']])
        modules = {'development_worlds': SimpleNamespace(catalog=SimpleNamespace(config=lambda world: config)),
                   'development_worlds.worlds': SimpleNamespace(DevelopmentSimulator=Reference),
                   'rift': SimpleNamespace(load_law=lambda path: SimpleNamespace(params={}, acceleration=candidate))}
        public = {'background_initial_state': {'ids': ['B'], 'positions': [[0., 0.]], 'velocities': [[1., 0.]]},
                  'source_property_target_id': None}
        with patch.dict(sys.modules, modules):
            outcome = recovery(2, 'velocity_dependent', public, count=60)
        self.assertFalse(outcome['correct'])


if __name__ == '__main__':unittest.main()
