"""Public world identifiers only; physical definitions live in shore_sim.

Neutral labels let the console select a world without giving discovery agents
any hint about its force law. Existing archives belong to world_1.
"""

DEFAULT_WORLD_ID = 'world_1'
WORLD_IDS = tuple(f'world_{index}' for index in range(1, 8))
LEGACY_PHYSICS_VERSION = 'v1'
CURRENT_PHYSICS_VERSION = 'v4'
PHYSICS_VERSIONS = (LEGACY_PHYSICS_VERSION, 'v2', 'v3', CURRENT_PHYSICS_VERSION)
WORLD_SETS = ('legacy', 'v15')
V15_PHYSICS_VERSION = 'shore_v15'
V15_WORLD_IDS = WORLD_IDS[:6]


def validate_world_set(value):
    if not isinstance(value, str) or value not in WORLD_SETS:
        raise ValueError('invalid_world_set')
    return value


def world_set_name(value):
    return {'legacy': '旧版', 'v15': '新版 · v15'}[validate_world_set(value)]


def validate_world_id(value, world_set='legacy'):
    ids = V15_WORLD_IDS if validate_world_set(world_set) == 'v15' else WORLD_IDS
    if not isinstance(value, str) or value not in ids:
        raise ValueError('invalid_world_id')
    return value


def validate_physics_version(value):
    if not isinstance(value, str) or value not in (*PHYSICS_VERSIONS, V15_PHYSICS_VERSION):
        raise ValueError('invalid_physics_version')
    return value


def validate_world_physics(world_id, physics_version):
    physics_version = validate_physics_version(physics_version)
    if physics_version == V15_PHYSICS_VERSION:
        return validate_world_id(world_id, 'v15'), physics_version
    world_id = validate_world_id(world_id)
    if world_id in ('world_6', 'world_7') and physics_version != 'v4':
        raise ValueError('world_physics_version_mismatch')
    return world_id, physics_version


def world_name(world_id, world_set='legacy'):
    return f'世界 {WORLD_IDS.index(validate_world_id(world_id, world_set)) + 1:02d}'


def public_worlds(world_set='legacy'):
    ids = V15_WORLD_IDS if validate_world_set(world_set) == 'v15' else WORLD_IDS
    return [{'id': world_id, 'name': world_name(world_id, world_set)} for world_id in ids]


def public_world_sets():
    return [{'id': key, 'name': world_set_name(key), 'worlds': public_worlds(key)}
            for key in WORLD_SETS]
