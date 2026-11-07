"""Deployment flag precedence; runs without importing rift or model packages."""
import argparse
from contextlib import redirect_stderr
import io
import unittest

from .pipeline_bridge.filter_options import add_filter_options, resolve_filter_enabled, resolve_filter_policy


class FilterOptionsTests(unittest.TestCase):
    def parser(self):
        parser = argparse.ArgumentParser()
        add_filter_options(parser)
        return parser

    def test_default_disabled_and_config_unchanged(self):
        config = {}
        self.assertIsNone(self.parser().parse_args([]).trajectory_filter)
        self.assertIs(resolve_filter_enabled(config), False)
        self.assertEqual(config, {})

    def test_all_precedence_combinations(self):
        parser = self.parser()
        for config in [{}, {'trajectory_filter_enabled': False}, {'trajectory_filter_enabled': True}]:
            for argv, override in [([], None), (['--trajectory-filter'], True), (['--no-trajectory-filter'], False)]:
                with self.subTest(config=config, argv=argv):
                    parsed = parser.parse_args(argv).trajectory_filter
                    self.assertIs(parsed, override)
                    expected = config.get('trajectory_filter_enabled', False) if override is None else override
                    self.assertIs(resolve_filter_enabled(config, parsed), expected)

    def test_invalid_config_rejected_even_with_override(self):
        for invalid in ['true', 'false', 0, 1, None, [], {}]:
            for override in [None, False, True]:
                with self.subTest(value=invalid, override=override):
                    with self.assertRaisesRegex(ValueError, 'must be boolean'):
                        resolve_filter_enabled({'trajectory_filter_enabled': invalid}, override)

    def test_invalid_override_or_numerics(self):
        for invalid in ['true', 0, 1, [], {}]:
            with self.assertRaises(ValueError):resolve_filter_enabled({}, invalid)
        for invalid in [None, [], 'config', 1]:
            with self.assertRaisesRegex(ValueError, 'numerics must be an object'):
                resolve_filter_enabled(invalid)

    def test_cli_flags_mutually_exclusive(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parser().parse_args(['--trajectory-filter', '--no-trajectory-filter'])

    def test_policy_default_and_cli_precedence(self):
        for configured in ['explicit', 'automatic']:
            for override in [None, 'explicit', 'automatic']:
                self.assertEqual(resolve_filter_policy({'trajectory_filter_policy': configured}, override), configured if override is None else override)
        self.assertEqual(resolve_filter_policy({}), 'explicit')
        args = self.parser().parse_args(['--trajectory-filter-policy', 'automatic'])
        self.assertEqual(args.trajectory_filter_policy, 'automatic')
        self.assertIs(resolve_filter_enabled({}, args.trajectory_filter), False)

    def test_invalid_policy_even_when_overridden(self):
        for value in [None, 1, True, [], {}, '', 'auto']:
            with self.assertRaises(ValueError):resolve_filter_policy({'trajectory_filter_policy': value}, 'explicit')
            if value is not None:
                with self.assertRaises(ValueError):resolve_filter_policy({}, value)
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):self.parser().parse_args(['--trajectory-filter-policy', 'auto'])


if __name__ == '__main__':unittest.main()
