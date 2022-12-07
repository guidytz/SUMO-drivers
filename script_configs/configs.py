from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from typing import Type, TypeVar


@dataclass(frozen=True)
class _Group:
    name: str = field(default="Common arguments")
    description: str = field(default="Params used in any simulation")

    def __eq__(self, __o: _Group) -> bool:
        return self.name == __o.name


def describe(text: str, shorten: bool = False, rename: str | None = None, group: _Group | None = None) -> dict:
    group = group or _Group()
    return dict(description=text, shorten=shorten, rename=rename, group=group)


def add_fields(parser: argparse.ArgumentParser, config: QLConfig | PQLConfig | NonLearnerConfig) -> argparse.ArgumentParser:
    current_group = _Group()
    group = parser.add_argument_group(current_group.name, current_group.description)
    group_map = {current_group: group}
    for argument, default in asdict(config).items():
        config_group = config.group(argument)
        name = config.rename(argument) or argument
        arg_names = [f"--{name}"]
        if config.shorten(argument):
            arg_names = [f"-{name[0]}"] + arg_names

        action = "store_true" if type(default) == bool else "store"
        required = default is None

        group = group_map.get(config_group)
        if group is None:
            group = parser.add_argument_group(config_group.name, config_group.description)
            group_map[config_group] = group

        arg = group.add_argument(*arg_names, default=default, dest=f"{argument}", action=action, required=required,
                                 help=config.description(argument),)

        if type(default) == list:
            arg.nargs = '+'

    return parser


T = TypeVar('T', bound='BaseConfig')


@dataclass(frozen=True)
class BaseConfig(ABC):
    sumocfg: str | None = field(default=None, metadata=describe("Path to '.sumocfg' file. MANDATORY"))
    demand: int = field(default=750, metadata=describe("Desired network demand.", shorten=True))
    steps: int = field(default=60000, metadata=describe("Number of max simulation steps.", shorten=True))
    wav: int = field(default=1, metadata=describe("Average in data collection window size.", shorten=True))
    gui: bool = field(default=False, metadata=describe("Use SUMO GUI flag.", shorten=True))
    nruns: int = field(default=1, metadata=describe("Number of multiple simulation runs.", shorten=True))
    parallel: int = field(default=False, metadata=describe(
        "Flag to indicate parallel runs with multiple simulations."))
    observe_list: list[str] = field(default_factory=lambda: ["TravelTime"],
                                    metadata=describe("Parameters to collect data from.", rename="observe-list"))

    def description(self, field_name: str) -> str:
        return self.__dataclass_fields__[field_name].metadata["description"]

    def shorten(self, field_name: str) -> bool:
        return self.__dataclass_fields__[field_name].metadata["shorten"]

    def rename(self, field_name) -> str | None:
        return self.__dataclass_fields__[field_name].metadata["rename"]

    def group(self, field_name) -> _Group:
        return self.__dataclass_fields__[field_name].metadata["group"]

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
class NonLearnerConfig(BaseConfig):
    """Base Non Learning Agent Simulation
    """

    @property
    def name(self) -> str:
        return "nl"


@dataclass(frozen=True)
class LearningAgentConfig(BaseConfig):
    """Base Learning Agent configs"""
    @staticmethod
    def main_group() -> _Group:
        return _Group(name="Learning Agent Params", description="Params used with any learning agent simulation")

    wait_learn: int = field(default=3000, metadata=describe(
        "Time steps to wait before the learning starts.", rename="wait-learn", group=main_group()))
    normalize_rewards: bool = field(default=False, metadata=describe(
        "Flag that indicates if rewards should be normalized. Requires a previous reward collection run.",
        rename="normalize-rewards", group=main_group()))
    collect_rewards: bool = field(default=False, metadata=describe(
        "Flag that indicates if rewards received should be collected to use them in normalizer in a posterior run.",
        rename="collect-rewards", group=main_group()))
    toll_speed: float = field(default=-1, metadata=describe("Speed threshold in which links should impose a toll on emission. "
                                                            "This parameter is only used in emission objectives. "
                                                            "The default indicates the toll is not used.",
                                                            group=main_group(), rename="toll-speed"))
    toll_value: float = field(default=-1, metadata=describe("Toll value to be added as penalty to emission. "
                                                            "This parameter is only used in emission objectives. "
                                                            "The default indicates the toll is not used.",
                                                            group=main_group(), rename="toll-value"))
    success_rate: float = field(default=0.0, metadata=describe("Communication success rate.", rename="success-rate",
                                                               group=_Group(name="Communication params", description="Params used when using C2I communication.")))
    queue_size: int = field(default=30, metadata=describe("CommDev queue size to store rewards.", rename="queue-size",
                                                          group=_Group(name="Communication params", description="Params used when using C2I communication.")))


@dataclass(frozen=True)
class QLConfig(LearningAgentConfig):
    """Base Q-Learning Agent Simulation
    """
    @staticmethod
    def main_group() -> _Group:
        return _Group(name="Q-Learning Agent Params", description="Params used with Q-Learning simulation")

    alpha: float = field(default=0.5, metadata=describe("Agent's learning rate.", group=main_group()))
    gamma: float = field(default=0.9, metadata=describe(
        "Agent's discount factor for future actions.", group=main_group()))
    bonus: int = field(default=500, metadata=describe(
        "Right destination bonus.", shorten=True, group=LearningAgentConfig.main_group()))
    penalty: int = field(default=500, metadata=describe(
        "Wrong destination penalty.", shorten=True, group=LearningAgentConfig.main_group()))
    objective: str = field(default="TravelTime", metadata=describe(
        "Agent's main objective to optimize", shorten=True, group=main_group()))

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

    gamma: float = field(default=0.9, metadata=describe(
        "Agent's discount factor for future actions.", group=main_group()))
    bonus: int = field(default=1, metadata=describe(
        "Right destination bonus.", shorten=True, group=LearningAgentConfig.main_group()))
    penalty: int = field(default=1, metadata=describe(
        "Wrong destination penalty.", shorten=True, group=LearningAgentConfig.main_group()))
    normalize_rewards: bool = field(default=True, metadata=describe(
        "Flag that indicates if rewards should be normalized. Requires a previous reward collection run.",
        rename="normalize-rewards", group=LearningAgentConfig.main_group()))
    objectives: list[str] = field(default_factory=lambda: ["TravelTime", "CO"], metadata=describe(
        "Agent's main objectives to optimize", shorten=True, group=main_group()))

    @property
    def name(self) -> str:
        return "pql"
