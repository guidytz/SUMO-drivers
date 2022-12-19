import sys

from script_configs import create_parser
from script_configs.configs import NonLearnerConfig, PQLConfig, QLConfig


def main(command_line=None):
    parser = create_parser()
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
