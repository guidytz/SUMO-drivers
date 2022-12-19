import argparse
import sys

from script_configs import CustomParser
from script_configs.configs import (NonLearnerConfig, PQLConfig, QLConfig,
                                    add_fields)


def main(command_line=None):
    parser = CustomParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars="@")
    subparsers = parser.add_subparsers(dest="command", help="Possible agent types")

    ql_base_config = QLConfig()
    ql = subparsers.add_parser(ql_base_config.name, help=ql_base_config.__doc__,
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ql = add_fields(ql, ql_base_config)
    ql.set_defaults(func=QLConfig.from_namespace)

    nl_base_config = NonLearnerConfig()
    nl = subparsers.add_parser(nl_base_config.name, help=nl_base_config.__doc__,
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    nl = add_fields(nl, nl_base_config)
    nl.set_defaults(func=NonLearnerConfig.from_namespace)

    pql_base_config = PQLConfig()
    pql = subparsers.add_parser(pql_base_config.name, help=pql_base_config.__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pql = add_fields(pql, pql_base_config)
    pql.set_defaults(func=PQLConfig.from_namespace)

    options = parser.parse_args(command_line)

    try:
        config: QLConfig | NonLearnerConfig | PQLConfig = options.func(options)
    except AttributeError:
        print("Wrong usage of script")
        parser.print_help()
        sys.exit(1)

    print(config.__dict__)


if __name__ == "__main__":
    main()
