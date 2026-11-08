"""Fast formula/interface regression tests; full numerical suite has separate CLI."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from development_worlds import catalog
from development_worlds.console import DevelopmentConsole
from development_worlds.evaluation import claims_truth, mission, _case_input, _validate_cases
from development_worlds.oracles import oracle_source
from development_worlds.worlds import DevelopmentSimulator, PARAMETERS, SINGLE_CENTER, WORLDS
from development_worlds.validate_suite import _upstream_module
from rift.law import load_law
from rift.prediction import rollout
ROOT = Path(__file__).resolve().parents[1]

class WorldSuiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='shore-law-test-')
        self.addCleanup(self.tmp.cleanup)

    def law(self, world):
        path = Path(self.tmp.name) / ('world_' + world + '.py')
        path.write_text(oracle_source(world), encoding='utf-8')
        return load_law(path)

    def test_twenty_registered_worlds_have_valid_case_domains(self):
        self.assertEqual(WORLDS, tuple(str(i) for i in range(1, 21)))
        self.assertEqual(catalog.WORLDS, WORLDS)
        for world in WORLDS:
            with self.subTest(world=world):
                self.assertLessEqual(len(PARAMETERS[world]), 5)
                _validate_cases(world, catalog.validation_cases(world), mission(world))
                self.assertEqual(claims_truth(world), DevelopmentSimulator(world).claims_truth)

    def test_independent_oracle_at_arbitrary_points_and_times(self):
        rng = np.random.default_rng(3217)
        points = rng.uniform(-6., 6., (17, 2)); points[0] = 0.
        for world in WORLDS:
            model, sim = self.law(world), DevelopmentSimulator(world)
            initial = sim.background_initial_state()
            for t in (0., 2.17, 15.875):
                for strength in (.5, 2.):
                    with self.subTest(world=world, t=t, strength=strength):
                        scalar = world in SINGLE_CENTER
                        attrs = [{'p2' if scalar else 'mass': .5 + 1.5*i/16} for i in range(17)]
                        source = {'p1': strength} if scalar else {}
                        background = [{'id': ident, 'kind': 'background', 'pos': pos,
                                       'vel': vel, 'properties': source if ident == 'N0' else {}}
                                      for ident, pos, vel in zip(initial['ids'], initial['positions'], initial['velocities'])]
                        probes = [{'id': f'P{i}', 'kind': 'probe', 'pos': pos, 'vel': [0., 0.], 'properties': attr}
                                  for i, (pos, attr) in enumerate(zip(points, attrs))]
                        actual = model.acceleration(t, background+probes, model.params)
                        expected = sim.acceleration(t, points, source_properties=source, probe_properties=attrs)
                        np.testing.assert_allclose(actual[-17:], expected, atol=1e-13, rtol=1e-13)
                        self.assertTrue(np.isfinite(actual).all())
                        np.testing.assert_array_equal(actual[0], [0., 0.])
                        if world == '14': np.testing.assert_array_equal(actual[1], [0., 0.])

    def test_real_candidate_interface_at_delayed_launch(self):
        for world in WORLDS:
            with self.subTest(world=world):
                case = copy.deepcopy(catalog.first_light(world))
                case['t_launch'] = .2; case['readout'] = [0., .05, .15, .4]
                expected = DevelopmentSimulator(world).launch(**case)
                actual = rollout(self.law(world), _case_input(mission(world), case))
                np.testing.assert_allclose(actual['positions'], expected['positions'], atol=2e-12, rtol=2e-12)
                np.testing.assert_allclose(actual['velocities'], expected['velocities'], atol=2e-12, rtol=2e-12)

    def test_original_six_short_trajectories_are_exact(self):
        module = _upstream_module('worlds')
        for world in WORLDS[:6]:
            with self.subTest(world=world):
                case = catalog.first_light(world)
                a = module.DevelopmentSimulator(world).launch(**case)
                b = DevelopmentSimulator(world).launch(**case)
                for key in ('positions', 'velocities', 'times'): np.testing.assert_array_equal(a[key], b[key])

    def test_new_worlds_finite_at_origin_and_source_centers(self):
        points = [[0., 0.], [-2., 0.], [2., 0.], [1.1, -.9], [1.2, -.7]]
        for world in WORLDS[6:]:
            with self.subTest(world=world):
                cfg = catalog.config(world)
                value = DevelopmentSimulator(world).acceleration(4.21, points,
                    source_properties={k: v['default'] for k, v in cfg['source_properties'].items()},
                    probe_properties={k: v['default'] for k, v in cfg['probe_properties'].items()})
                self.assertTrue(np.isfinite(value).all())

    def test_controlled_source_and_inertia_scaling(self):
        points = [[.7, -1.3], [3.1, 2.2]]
        for world in SINGLE_CENTER:
            with self.subTest(world=world):
                sim = DevelopmentSimulator(world)
                base = sim.acceleration(2.3, points, source_properties={'p1': 1.}, probe_properties={'p2': 1.})
                scaled = sim.acceleration(2.3, points, source_properties={'p1': 2.}, probe_properties={'p2': .5})
                np.testing.assert_allclose(scaled, 4*base, atol=0., rtol=1e-15)

    def test_time_dependence_labels_match_force(self):
        for world in WORLDS:
            with self.subTest(world=world):
                cfg = catalog.config(world)
                kwargs = {'source_properties': {k: v['default'] for k, v in cfg['source_properties'].items()},
                          'probe_properties': {k: v['default'] for k, v in cfg['probe_properties'].items()}}
                sim = DevelopmentSimulator(world)
                a = sim.acceleration(0., [[1.25, -.7], [-2., 3.]], **kwargs)
                b = sim.acceleration(2.345, [[1.25, -.7], [-2., 3.]], **kwargs)
                self.assertEqual(not np.array_equal(a, b), claims_truth(world)['time_dependence'])

    def test_console_seed_budget_and_archive(self):
        for world in WORLDS:
            with self.subTest(world=world):
                cfg = catalog.config(world)
                a = DevelopmentConsole(world, catalog.noise_sigma(world), 42, cfg, clock=lambda: 0.)
                b = DevelopmentConsole(world, catalog.noise_sigma(world), 42, cfg, clock=lambda: 0.)
                self.assertEqual(a.mission(), b.mission())
                self.assertEqual(a.budget['energy_used'], 0)
                request = catalog.first_light(world)
                self.assertEqual(a.launch(**request), b.launch(**request))
                self.assertEqual(a.budget['energy_used'], 2)
                self.assertEqual(a.archive(), a.archive())

    def test_illegal_requests_do_not_spend_energy(self):
        for world in ('7', '14', '20'):
            with self.subTest(world=world):
                console = DevelopmentConsole(world, catalog.noise_sigma(world), 0, catalog.config(world), clock=lambda: 0.)
                console.mission()
                request = catalog.first_light(world); request['readout'] = [.007]
                with self.assertRaises(ValueError): console.launch(**request)
                self.assertEqual(console.budget['energy_used'], 0)
                request = catalog.first_light(world); request['probes'][0]['pos'] = [100., 0.]
                with self.assertRaises(ValueError): console.launch(**request)
                self.assertEqual(console.budget['energy_used'], 0)

if __name__ == '__main__': unittest.main()
