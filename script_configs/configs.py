from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from typing import Type, TypeVar


def describe(text: str, shorten: bool = False) -> dict:
    return dict(description=text, shorten=shorten)


def add_fields(parser: argparse.ArgumentParser, config: QLConfig) -> argparse.ArgumentParser:
    for argument, default in asdict(config).items():
        arg_names = [f"--{argument}"]
        if config.shorten(argument):
            arg_names = [f"-{argument[0]}"] + arg_names

        action = "store_true" if type(default) == bool else "store"
        required = default is None

        parser.add_argument(*arg_names, default=default, dest=f"{argument}", action=action, required=required,
                            help=config.description(argument))

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

    def description(self, field_name: str) -> str:
        return self.__dataclass_fields__[field_name].metadata["description"]

    def shorten(self, field_name: str) -> bool:
        return self.__dataclass_fields__[field_name].metadata["shorten"]

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError("This method should be called from a concrete class")

    @classmethod
    def from_namespace(cls: Type[T], args: argparse.Namespace) -> T:
        params_dict = args.__dict__
        del params_dict["command"]
        return cls(**params_dict)


@dataclass(frozen=True)
class QLConfig(BaseConfig):
    """Base Q-Learning Agent.
    """

    alpha: float = field(default=0.5, metadata=describe("Agent's learning rate."))
    gamma: float = field(default=0.9, metadata=describe("Agent's discount factor for future actions."))

    @property
    def name(self) -> str:
        return "ql"
