import argparse
import sys

from script_configs.configs import QLConfig, add_fields


def main(command_line=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", help="Possible agent types")

    ql_base_config = QLConfig()
    ql = subparsers.add_parser(ql_base_config.name, help=QLConfig.__doc__,
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ql = add_fields(ql, ql_base_config)

    # pql = subparsers.add_parser("pql", help="bla", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # no_learn = subparsers.add_parser("nl", help="bla", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = parser.parse_args(command_line)
    config: QLConfig
    match args.command:
        case ql_base_config.name:
            config = QLConfig.from_namespace(args)
        case _:
            print("Unknown param.")
            sys.exit(1)
    print(config.__dict__)


if __name__ == "__main__":
    main()
