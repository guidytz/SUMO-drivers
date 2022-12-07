import argparse
import sys

from script_configs.configs import NonLearnerConfig, QLConfig, add_fields


def main(command_line=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", help="Possible agent types")

    ql_base_config = QLConfig()
    ql = subparsers.add_parser(ql_base_config.name, help=QLConfig.__doc__,
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ql = add_fields(ql, ql_base_config)
    ql.set_defaults(func=QLConfig.from_namespace)

    nl_base_config = NonLearnerConfig()
    nl = subparsers.add_parser(nl_base_config.name, help=NonLearnerConfig.__doc__,
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    nl = add_fields(nl, nl_base_config)
    nl.set_defaults(func=NonLearnerConfig.from_namespace)

    # pql = subparsers.add_parser("pql", help="bla", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # no_learn = subparsers.add_parser("nl", help="bla", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = parser.parse_args(command_line)

    try:
        config: QLConfig | NonLearnerConfig = args.func(args)
    except AttributeError:
        print("Wrong usage of script")
        parser.print_help()
        sys.exit(1)

    print(config.__dict__)


if __name__ == "__main__":
    main()
