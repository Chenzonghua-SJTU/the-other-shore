"""Pure deployment-option parsing, independent of models and the official SDK."""


def add_filter_options(parser):
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument('--trajectory-filter', dest='trajectory_filter', action='store_true',
                       help='Enable optional joint spline tool, overriding agent config')
    flags.add_argument('--no-trajectory-filter', dest='trajectory_filter', action='store_false',
                       help='Disable optional joint spline tool, overriding agent config')
    parser.set_defaults(trajectory_filter=None)
    parser.add_argument('--trajectory-filter-policy', choices=('explicit', 'automatic'), default=None,
                        help='Filter policy override; does not enable filtering by itself')


def resolve_filter_enabled(numerics, cli_override=None):
    """Validate configuration even when overridden; missing key means False."""
    if not isinstance(numerics, dict):
        raise ValueError('numerics must be an object')
    configured = numerics.get('trajectory_filter_enabled', False)
    if type(configured) is not bool:
        raise ValueError('numerics.trajectory_filter_enabled must be boolean')
    if cli_override is not None and type(cli_override) is not bool:
        raise ValueError('trajectory filter CLI override must be boolean or None')
    return configured if cli_override is None else cli_override


def resolve_filter_policy(numerics, cli_override=None):
    if not isinstance(numerics, dict):
        raise ValueError('numerics must be an object')
    configured = numerics.get('trajectory_filter_policy', 'explicit')
    if not isinstance(configured, str) or configured not in ('explicit', 'automatic'):
        raise ValueError('numerics.trajectory_filter_policy must be explicit or automatic')
    if cli_override is not None and (not isinstance(cli_override, str) or cli_override not in ('explicit', 'automatic')):
        raise ValueError('trajectory filter CLI policy must be explicit or automatic')
    return configured if cli_override is None else cli_override
