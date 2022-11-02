from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np
from gym import spaces
from sumo_ql.exploration.epsilon_greedy import EpsilonGreedy


class Agent(ABC):
    def __init__(self, action_space: Dict[str, spaces.Discrete],
                 alpha: float,
                 gamma: float,
                 exploration_strategy: EpsilonGreedy) -> None:
        """Class constructor

        Args:
            action_space (dict): dictionary containing all gym spaces regarding actions available in each state
            alpha (float, optional): learning rate parameter in Q-Learning. Defaults to 0.5.
            gamma (float, optional): discount factor for future rewards in Q-Learning. Defaults to 0.9.
            exploration_strategy (EpsilonGreedy, optional): exploration strategy used. Defaults to EpsilonGreedy().
        """
        self._action_space: Dict[str, spaces.Discrete] = action_space
        self._alpha: float = alpha
        self._gamma: float = gamma
        self._exploration_strategy = exploration_strategy

    @abstractmethod
    def act(self, state, available_actions) -> int:
        """Method that performs, i.e. chooses, a new action for the agent.

        Args:
            state (str): the state the agent is currently in.
            available_actions (List[int]): A list containing the available actions for the given state at the moment.

        Returns:
            int: the action chosen in the agent's current state.
        """
        raise NotImplementedError

    @abstractmethod
    def learn(self, action, current_state, next_state, reward) -> None:
        """Method that updates the Q-table based on the reward received for the action taken at the given current state,


        Args:
            action (int): The action taken.
            current_state (str): The state where the action was taken.
            next_state (str): The next state reached with the action.
            reward (float, np.array): The reward received for taking the action within the given state.

        Raises:
            RuntimeError: the method raises a RuntimeError if the new state given is not in Q-table
        """
        raise NotImplementedError


class QLAgent(Agent):
    """Class responsible for handling the learning agents that control the vehicles route building.
    """

    def __init__(self, action_space: Dict[str, spaces.Discrete],
                 alpha: float = 0.5,
                 gamma: float = 0.9,
                 exploration_strategy: EpsilonGreedy = EpsilonGreedy()) -> None:
        """Class constructor

        Args:
            action_space (dict): dictionary containing all gym spaces regarding actions available in each state
            alpha (float, optional): learning rate parameter in Q-Learning. Defaults to 0.5.
            gamma (float, optional): discount factor for future rewards in Q-Learning. Defaults to 0.9.
            exploration_strategy (EpsilonGreedy, optional): exploration strategy used. Defaults to EpsilonGreedy().
        """
        super().__init__(action_space, alpha, gamma, exploration_strategy)
        self.__q_table = {state: [0. for _ in range(self._action_space[state].n)]
                          for state in self._action_space.keys()}

    def act(self, state: str, available_actions: list[int]) -> int:
        match self._exploration_strategy.choose(self.__q_table, state, self._action_space, available_actions):
            case int(result):
                return result
            case _:
                raise RuntimeError("Something went wrong exploration strategy should return an integer!")

    def learn(self, action: int, current_state: str, next_state: str, reward: float) -> None:
        if next_state not in self.__q_table:
            raise RuntimeError("Invalid state: not found in Q-table!")

        max_future_value = max(self.__q_table[next_state])
        self.__q_table[current_state][action] += self._alpha * (reward + self._gamma * max_future_value -
                                                                self.__q_table[current_state][action])


class PQLAgent(Agent):
    def __init__(self, action_space: Dict[str, spaces.Discrete],
                 alpha: float = 0.5,
                 gamma: float = 0.9,
                 exploration_strategy: EpsilonGreedy = EpsilonGreedy(),
                 n_objectives: int = 2) -> None:
        """Class constructor

        Args:
            action_space (dict): dictionary containing all gym spaces regarding actions available in each state
            alpha (float, optional): learning rate parameter in Q-Learning. Defaults to 0.5.
            gamma (float, optional): discount factor for future rewards in Q-Learning. Defaults to 0.9.
            exploration_strategy (EpsilonGreedy, optional): exploration strategy used. Defaults to EpsilonGreedy().
        """
        super().__init__(action_space, alpha, gamma, exploration_strategy)
        self.__n_obj = n_objectives
        self.__non_dominated = {state: [[np.zeros(self.__n_obj)] for _ in range(self._action_space[state].n)]
                                for state in self._action_space.keys()}
        self.__avg_rewards = {state: [np.zeros(self.__n_obj) for _ in range(self._action_space[state].n)]
                              for state in self._action_space.keys()}
        self.__visits = {state: [0 for _ in range(self._action_space[state].n)] for state in self._action_space.keys()}

    def act(self, state: str, available_actions: list[int]) -> tuple[int, int]:
        q_set = self.__compute_q_set(state)
        match self._exploration_strategy.choose(q_set, state, self._action_space, available_actions):
            case int(action), int(chosen_obj):
                return action, chosen_obj
            case _:
                raise RuntimeError("Something went wrong, exploration strategy should return a tuple.")

    def learn(self, action: int, current_state: str, next_state: str, reward: np.ndarray) -> None:
        self.__update_nd(current_state, action, next_state)

        self.__visits[current_state][action] += 1
        self.__avg_rewards[current_state][action] += (
            (reward - self.__avg_rewards[current_state][action]) / self.__visits[current_state][action]
        )

    def get_non_dominated(self, state: str, action: int):
        return self.__non_dominated[state][action]

    def __q_set(self, state: str, action: int) -> list[np.ndarray]:
        non_dom = self.__non_dominated[state][action]
        return [self.__avg_rewards[state][action] + self._gamma * nd for nd in non_dom]

    def __compute_q_set(self, state: str) -> np.ndarray:
        return np.array([item for action in range(self._action_space[state].n) for item in self.__q_set(state, action)])

    def __update_nd(self, curr_state: str, action: int, next_state: str) -> None:
        next_st_q_set = self.__compute_q_set(next_state)
        self.__non_dominated[curr_state][action] = self.__pareto_nd(next_st_q_set)

    def __pareto_nd(self, solutions: np.ndarray) -> list[np.ndarray]:
        is_efficient = np.ones(solutions.shape[0], dtype=bool)
        for i, val in enumerate(solutions):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(solutions[is_efficient] > val, axis=1)
                is_efficient[i] = 1

        return list(solutions[is_efficient])
