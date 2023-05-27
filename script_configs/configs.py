from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type, TypeAlias, TypeVar


@dataclass(frozen=True)
class _Group:
    name: str = field(default="Common arguments")
    description: str = field(default="Params used in any simulation")

    def __eq__(self, __o: _Group) -> bool:
        return self.name == __o.name


def describe(text: str, shorten: bool = False, rename: str | None = None, group: _Group | None = None) -> dict:
    group = group or _Group()
    return dict(description=text, shorten=shorten, rename=rename, group=group)


T = TypeVar('T', bound='EmptyConfig')


@dataclass(frozen=True)
class EmptyConfig(ABC):
    def description(self, field_name: str) -> str:
        return self.__dataclass_fields__[field_name].metadata["description"]

    def shorten(self, field_name: str) -> bool:
        return self.__dataclass_fields__[field_name].metadata["shorten"]

    def rename(self, field_name) -> str | None:
        return self.__dataclass_fields__[field_name].metadata["rename"]

    def group(self, field_name) -> _Group:
        return self.__dataclass_fields__[field_name].metadata["group"]

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError("This method should be called from a concrete class")

    @staticmethod
    def main_group() -> _Group:
        return _Group()

    @classmethod
    def from_namespace(cls: Type[T], args: argparse.Namespace) -> T:
        params_dict = args.__dict__
        del params_dict["command"]
        del params_dict["func"]
        return cls(**params_dict)


@dataclass(frozen=True)
class BaseConfig(EmptyConfig):
    sumocfg: str | None = field(default=None, metadata=describe("Path to '.sumocfg' file. MANDATORY"))
    steps: int = field(default=60000, metadata=describe("Number of max simulation steps.", shorten=True))
    demand: int = field(default=750, metadata=describe("Desired network demand.", shorten=True))
    aw: int = field(default=1, metadata=describe("Average in data collection window size.", shorten=True))
    gui: bool = field(default=False, metadata=describe("Use SUMO GUI flag.", shorten=True))
    nruns: int = field(default=1, metadata=describe("Number of multiple simulation runs.", shorten=True))
    parallel: bool = field(default=False,
                           metadata=describe("Flag to indicate parallel runs with multiple simulations."))

    observe_list: list[str] = field(default_factory=lambda: ["TravelTime"],
                                    metadata=describe("Parameters to collect data from.", rename="observe-list"))


@dataclass(frozen=True)
class NonLearnerConfig(BaseConfig):
    """Base Non Learning Agent Simulation
    """

    @property
    def name(self) -> str:
        return "nl"


@dataclass(frozen=True)
class GraphConfig(EmptyConfig):
    """Virtual Graph config with its params
    """

    @staticmethod
    def main_group() -> _Group:
        return _Group(name="Graph Params (Used with communication)", description="Params used virtual graph")

    file: str | None = field(default=None,
                             metadata=describe("Path to csv file that will be used as input for the virtual graph.", 
                                               rename="vg-file", group=main_group()))

    attributes: list[str] = field(default_factory=lambda: ["ALL"],
                                  metadata=describe("List of attributes used to create the virtual graph. "
                                                    "Attribute is given by the number of the column of the input csv.",
                                                    rename="vg-attributes", group=main_group()))

    labels: list[str | None] = field(default_factory=lambda: [None],
                                    metadata=describe("List of attributes that will compose the label of each vertex in the virtual graph. "
                                                    "Attribute is given by the number of the column of the input csv. The first attribute "
                                                    "passed will determine which attribute is used to aggregate the virtual graph neighbors, "
                                                    "i.e. aggregate by link or junction.", rename="vg-label", group=main_group()))

    restrictions: list[str] | None = field(default_factory=lambda: None,
                                   metadata=describe("List of attributes that the vertices cannot share in order to create an edge in the virtual "
                                                     "graph. Attribute is given by the number of the column of the input csv.", rename="vg-restrictions",
                                                     group=main_group()))

    threshold: float = field(default=0., metadata=describe("Threshold used to create an edge in the virtual graph.",
                                                           rename="vg-threshold", group=main_group()))

    use_or: bool = field(default=False, metadata=describe("Flag that indicates or logic instead of the and logic to create an edge between vertices given "
                                                          "multiple attributes.",
                                                          rename="use-or-logic", group=main_group()))

    measures: list[str] | None = field(default_factory=lambda: None,
                                metadata=describe("List of centrality measures to be taken of the virtual graph.",
                                                  rename="centrality-measures", group=main_group()))

    no_image: bool = field(default=False, metadata=describe("Flag to indicate to the script not to generate the virtual graph image.", 
                                                            rename="no-image", group=main_group()))

    raw: bool = field(default=False, metadata=describe("Flag to indicate not to remove vertices with degree zero.",
                                                       rename="raw-graph", group=main_group()))

    giant: bool = field(default=False, metadata=describe("Flag to indicate that only the giant component of the virtual graph should be "
                                                         "presented in its image.", group=main_group()))

    not_normalize: bool = field(default=False, metadata=describe("Flag to indicate to the script not to normalize the input csv data to generate "
                                                                 "the virtual graph.", rename="vg-not-normalize", group=main_group()))

    min_degree: int = field(default=0, metadata=describe("Determines the minimum degree a vertex should have in order to be plotted in the virtual "
                                                         "graph image.", rename="min-degree", group=main_group()))

    min_step: int = field(default=0, metadata=describe("Determines the minimum step a vertex should have in order to be plotted in the virtual graph image.", 
                                                       rename="vg-min-step", group=main_group()))

    vg_dict: str | None = field(default=None, metadata=describe("Name of file containing python dictionary of virtual graph "
                                                       "neighbors. Only used to load virtual graph into simulation", 
                                                       rename="vg-dict-file", group=main_group()))

    interval: int = field(default=250, metadata=describe("Amplitude of the timestep interval of the virtual graph "
                                                         "neighbors dictionary.", group=main_group()))

    @property
    def name(self) -> str:
        return "graph"

    @classmethod
    def create(cls, params: dict) -> None | GraphConfig:
        dict_file = params.get("vg_dict")
        if dict_file is not None:
            return cls(**params)

        file = params.get("file")
        labels = params.get("labels")

        match file, labels:
            case None, [None]:
                return

            case None, _:
                raise ValueError("Graph file not informed!")

            case _, None:
                raise ValueError("Label is a necessary parameter for graphs!")

            case _, [None]:
                raise ValueError("Label is a necessary parameter for graphs!")

            case _:
                return cls(**params)


@dataclass(frozen=True)
class CommunicationConfig(EmptyConfig):
    """Communication config with its params
    """
    @staticmethod
    def main_group() -> _Group:
        return _Group(name="Communication Params", description="Params used with C2I communication")

    @property
    def name(self) -> str:
        return "comm"

    success_rate: float = field(default=0.0, metadata=describe("Communication success rate.", rename="success-rate",
                                                               group=main_group()))

    queue_size: int = field(default=30, metadata=describe("CommDev queue size to store rewards.", rename="queue-size",
                                                          group=main_group()))


@dataclass(frozen=True)
class LearningAgentConfig(BaseConfig):
    """Base Learning Agent configs"""
    @staticmethod
    def main_group() -> _Group:
        return _Group(name="Learning Agent Params", description="Params used with any learning agent simulation")

    wait_learn: int = field(default=3000, metadata=describe("Time steps to wait before the learning starts.",
                                                            rename="wait-learn", group=main_group()))

    normalize_rewards: bool = field(default=False, metadata=describe("Flag that indicates if rewards should be "
                                                                     "normalized. Requires a previous reward "
                                                                     "collection run.",
                                                                     rename="normalize-rewards", group=main_group()))

    collect_rewards: bool = field(default=False, metadata=describe("Flag that indicates if rewards received should be "
                                                                   "collected to use them in normalizer in a posterior "
                                                                   "run.",
                                                                   rename="collect-rewards", group=main_group()))

    toll_speed: float = field(default=-1, metadata=describe("Speed threshold in which links should impose a toll on "
                                                            "emission. This parameter is only used in emission "
                                                            "objectives. The default indicates the toll is not used.",
                                                            group=main_group(), rename="toll-speed"))

    toll_value: int = field(default=-1, metadata=describe("Toll value to be added as penalty to emission. "
                                                          "This parameter is only used in emission objectives. "
                                                          "The default indicates the toll is not used.",
                                                          group=main_group(), rename="toll-value"))

    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    virtual_graph: GraphConfig = field(default_factory=GraphConfig)

    @classmethod
    def from_namespace(cls: Type[T], args: argparse.Namespace) -> T:
        comm_config_dict = CommunicationConfig().__dict__
        graph_config_dict = GraphConfig().__dict__

        params_dict = args.__dict__.copy()
        for arg, value in params_dict.items():
            if arg in comm_config_dict:
                comm_config_dict[arg] = value
                del args.__dict__[arg]
            elif arg in graph_config_dict:
                graph_config_dict[arg] = value
                del args.__dict__[arg]

        main_config_dict = super().from_namespace(args).__dict__

        communication = CommunicationConfig(**comm_config_dict)
        main_config_dict["communication"] = communication

        graph = GraphConfig.create(graph_config_dict)
        main_config_dict["virtual_graph"] = graph

        return cls(**main_config_dict)


@dataclass(frozen=True)
class QLConfig(LearningAgentConfig):
    """Base Q-Learning Agent Simulation
    """
    @staticmethod
    def main_group() -> _Group:
        return _Group(name="Q-Learning Agent Params", description="Params used with Q-Learning simulation")

    alpha: float = field(default=0.5, metadata=describe("Agent's learning rate.", group=main_group()))
    gamma: float = field(default=0.9, metadata=describe("Agent's discount factor for future actions.",
                                                        group=main_group()))

    bonus: int = field(default=500, metadata=describe("Right destination bonus.", shorten=True,
                                                      group=LearningAgentConfig.main_group()))

    penalty: int = field(default=500, metadata=describe("Wrong destination penalty.", shorten=True,
                                                        group=LearningAgentConfig.main_group()))

    objective: str = field(default="TravelTime", metadata=describe("Agent's main objective to optimize", shorten=True,
                                                                   group=main_group()))

    @property
    def name(self) -> str:
        return "ql"


@dataclass(frozen=True)
class PQLConfig(LearningAgentConfig):
    """Base Pareto Q-Learning Agent Simulation
    """
    @staticmethod
    def main_group() -> _Group:
        return _Group(name="Pareto Q-Learning Agent Params", description="Params used with Pareto Q-Learning simulation")

    gamma: float = field(default=0.9, metadata=describe("Agent's discount factor for future actions.",
                                                        group=main_group()))

    bonus: int = field(default=1, metadata=describe("Right destination bonus.", shorten=True,
                                                    group=LearningAgentConfig.main_group()))

    penalty: int = field(default=1, metadata=describe("Wrong destination penalty.", shorten=True,
                                                      group=LearningAgentConfig.main_group()))

    normalize_rewards: bool = field(default=True, metadata=describe("Flag that indicates if rewards should be "
                                                                    "normalized. Requires a previous reward collection "
                                                                    "run.",
                                                                    rename="normalize-rewards",
                                                                    group=LearningAgentConfig.main_group()))

    objectives: list[str] = field(default_factory=lambda: ["TravelTime", "CO"],
                                  metadata=describe("Agent's main objectives to optimize", shorten=True,
                                                    group=main_group()))

    @property
    def name(self) -> str:
        return "pql"


Config: TypeAlias = QLConfig | PQLConfig | NonLearnerConfig | CommunicationConfig | GraphConfig


def add_fields(parser: argparse.ArgumentParser, config: Config) -> argparse.ArgumentParser:
    group_map = {}
    if issubclass(config.__class__, BaseConfig):
        current_group = _Group()
        group = parser.add_argument_group(current_group.name, current_group.description)
        group_map = {current_group: group}
    for argument, value in config:
        match value:
            case CommunicationConfig(_):
                add_fields(parser, value)
            case GraphConfig(_):
                add_fields(parser, value)
            case _:
                config_group = config.group(argument)
                name = config.rename(argument) or argument
                arg_names = [f"--{name}"]
                if config.shorten(argument):
                    arg_names = [f"-{name[0]}"] + arg_names

                action = "store_true" if type(value) == bool else "store"
                required = value is None and config.__class__ is not GraphConfig

                group = group_map.get(config_group)
                if group is None:
                    group = parser.add_argument_group(config_group.name, config_group.description)
                    group_map[config_group] = group

                arg = group.add_argument(*arg_names, default=value, dest=f"{argument}", action=action, required=required,
                                         help=config.description(argument))

                match value:
                    case None:
                        arg.type = str
                    case list(_):
                        arg.type = str
                    case _:
                        arg.type = type(value)

                if type(value) == list:
                    arg.nargs = '+'

    return parser
