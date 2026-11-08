"""Transport compatibility checks using the unmodified v15 client."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest

from development_worlds import catalog
from development_worlds.console import DevelopmentConsole
from development_worlds.service import DevelopmentSession, serve_session
from rift.console import Console, ConsoleError


class TransportTests(unittest.TestCase):
    def test_all_worlds_public_mission_and_budget(self):
        for world in catalog.WORLDS:
            with self.subTest(world=world):
                console = DevelopmentConsole(world, catalog.noise_sigma(world), 123,
                                             catalog.config(world), clock=lambda: 0.)
                mission = console.mission()
                self.assertEqual(mission['world_id'], world)
                self.assertEqual(console.budget['energy_used'], 0)
                self.assertEqual(len(mission['first_light']['positions']), 8)
                self.assertNotIn('claims_truth', mission)
                self.assertNotIn('parameters', mission)
                launch = copy.deepcopy(catalog.first_light(world))
                console.launch(**launch)
                self.assertEqual(console.budget['energy_used'], 2)
                archive = console.archive()
                self.assertEqual(archive['first_light'], mission['first_light'])

    def test_world20_http_roundtrip(self):
        console = DevelopmentConsole('20', catalog.noise_sigma('20'), 99,
                                     catalog.config('20'), clock=lambda: 0.)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = DevelopmentSession(console, root/'session')
            with serve_session(session) as credentials:
                client = Console(**credentials)
                mission = client.mission()
                self.assertEqual(mission['world_id'], '20')
                request = catalog.first_light('20')
                result = client.launch(**request)
                self.assertEqual(result['budget']['energy_used'], 2)
                alias_request = copy.deepcopy(request)
                alias_request['probes'] = alias_request['probes'][:1]
                alias_result = client._call('insert_probe', alias_request)
                self.assertEqual(alias_result['budget']['energy_used'], 3)
                self.assertEqual(client.observe([0., .1])['budget']['observe_used'], 1)
                self.assertEqual(client.archive()['first_light'], mission['first_light'])
                bad = copy.deepcopy(request)
                bad['probes'][0]['pos'] = [1000., 0.]
                with self.assertRaises(ConsoleError):
                    client.launch(**bad)
                self.assertEqual(console.budget['energy_used'], 3)
                # This is an arbitrary ABI smoke candidate, not a discovered law.
                law = root/'law.py'
                law.write_text('import numpy as np\nLAW_API_VERSION="1.0.0"\nPARAMS={}\nBODY_PROPS={}\nHIDDEN=[]\ndef acceleration(t,bodies,params):\n    return np.zeros((len(bodies),2))\n')
                claims = root/'claims.json'
                claims.write_text(json.dumps(dict(dynamics_order=2, hidden_matter=False, time_dependence=False)))
                saved = client.save_draft(law, claims)
                self.assertTrue(saved['accepted'])
                submitted = client.submit(law, claims)
                self.assertTrue(submitted['submitted'])
                with self.assertRaises(ConsoleError):
                    client.observe([.2])
                self.assertEqual(session.latest['law_sha256'], submitted['law_sha256'])


if __name__ == '__main__':
    unittest.main()
