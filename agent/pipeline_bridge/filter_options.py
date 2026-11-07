"""Pure deployment-option parsing, independent of models and the official SDK."""


def add_filter_options(parser):
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument('--trajectory-filter', dest='trajectory_filter', action='store_true',
                       help='Enable optional joint spline tool, overriding agent config')
    flags.add_argument('--no-trajectory-filter', dest='trajectory_filter', action='store_false',
                       help='Disable optional joint spline tool, overriding agent config')
    parser.set_defaults(trajectory_filter=None)


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
