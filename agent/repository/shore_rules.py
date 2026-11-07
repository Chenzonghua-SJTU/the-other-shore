"""Versioned local discovery rules; absent metadata always retains E3 semantics."""
from copy import deepcopy
import math


def rules_for(environment=None):
    """Return a fresh rules dict for an environment mapping (or ruleset name)."""
    name = environment if isinstance(environment, str) else (environment or {}).get('ruleset', 'e3')
    if name not in ('e3', 'e4', 'e5'):
        raise ValueError('unknown_ruleset')
    return {'ruleset': name, 'energy_budget': 10 if name in ('e4', 'e5') else 100,
            'wall_seconds': 300 if name in ('e4', 'e5') else 3600,
            'max_tokens_per_call': 4096 if name in ('e4', 'e5') else None,
            'score_formula_version': 'shore_e4_10energy' if name in ('e4', 'e5') else 'shore_pdf_1101'}


def apply_model_rules(model):
    """Enforce an explicitly tagged model profile at the last request boundary."""
    model = deepcopy(model)
    rules = rules_for(model)
    if rules['ruleset'] in ('e4', 'e5'):
        value = model.get('max_tokens_per_call', 4096)
        if type(value) is not int or value <= 0:
            raise ValueError('invalid_output_budget')
        model.update(thinking=False, max_tokens_per_call=min(value, 4096))
    return model


def apply_rules(config):
    """Copy a merged config and enforce E4/E5; E3 settings remain byte-value compatible."""
    config = deepcopy(config)
    environment = config.setdefault('environment', {})
    rules = rules_for(environment)
    if rules['ruleset'] in ('e4', 'e5'):
        if environment.get('world_set', 'v15') != 'v15':
            raise ValueError(rules['ruleset'] + '_requires_v15')
        environment.update(energy_budget=10, wall_seconds=300, noise=rules['ruleset'] == 'e4')
        model = config.setdefault('model', {})
        model['ruleset'] = rules['ruleset']
        config['model'] = apply_model_rules(model)
    return config


def initial_energy(mission=None, environment=None):
    """Read the actual mission limit first, falling back to the archive's ruleset."""
    mission = mission if isinstance(mission, dict) else {}
    mission = mission.get('data', mission)
    task = mission.get('task', {}) if isinstance(mission, dict) else {}
    limits = task.get('limits', {}) if isinstance(task, dict) else {}
    limits = limits if isinstance(limits, dict) else {}
    for key in ('initial_energy', 'energy_budget'):
        value = limits.get(key)
        if type(value) in (int, float) and math.isfinite(value) and value > 0:
            return value
    return rules_for(environment)['energy_budget']
