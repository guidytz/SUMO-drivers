from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from typing import Type, TypeVar

DEFAULT_GROUP = {"name": "Common arguments", "description": "Params used in any simulation"}


def describe(text: str, shorten: bool = False, rename: str | None = None, group: dict | None = None) -> dict:
    group = group or DEFAULT_GROUP
    return dict(description=text, shorten=shorten, rename=rename, group=group)


def add_fields(parser: argparse.ArgumentParser, config: QLConfig | NonLearnerConfig) -> argparse.ArgumentParser:
    current_group = DEFAULT_GROUP
    group = parser.add_argument_group(current_group["name"], current_group["description"])
    for argument, default in asdict(config).items():
        config_group = config.group(argument)
        name = config.rename(argument) or argument
        arg_names = [f"--{name}"]
        if config.shorten(argument):
            arg_names = [f"-{name[0]}"] + arg_names

        action = "store_true" if type(default) == bool else "store"
        required = default is None

        if config_group != current_group:
            group = parser.add_argument_group(config_group["name"], config_group["description"])
            current_group = config_group

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

    def group(self, field_name) -> dict:
        return self.__dataclass_fields__[field_name].metadata["group"]

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError("This method should be called from a concrete class")

    @staticmethod
    def main_group() -> dict:
        return DEFAULT_GROUP

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
class QLConfig(BaseConfig):
    """Base Q-Learning Agent Simulation
    """

    wait_learn: int = field(default=3000, metadata=describe(
        "Time steps to wait before the learning starts.", rename="wait-learn"))
    alpha: float = field(default=0.5, metadata=describe("Agent's learning rate."))
    gamma: float = field(default=0.9, metadata=describe("Agent's discount factor for future actions."))

    @staticmethod
    def main_group() -> dict:
        return {"name": "QLParams", "description": "Params used with Q-Learning simulation"}

    @property
    def name(self) -> str:
        return "ql"
