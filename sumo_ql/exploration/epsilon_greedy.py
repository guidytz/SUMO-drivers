import random as rd
from datetime import datetime
from typing import Dict, Union
from typing import List
from gym import spaces
import numpy as np

class EpsilonGreedy:
    """Class responsible for the exploration strategy in action choice.

    Args:
        initial_epsilon (float, optional): desired starting value for epsilon. Defaults to 1.0.
        min_epsilon (float, optional): minimum value accepted for epsilon after decay. Defaults to 0.05.
        decay (float, optional): decay rate for each time it makes a choice. Defaults to 0.99.
        seed (int, optional): seed to use in random choice. Defaults to datetime.now().
    """

    def __init__(self, initial_epsilon: float = 1.0,
                 min_epsilon: float = 0.05,
                 decay: float = 0.99,
                 seed: int = datetime.now()) -> None:
        self.__initial_epsilon = initial_epsilon
        self.__epsilon = initial_epsilon
        self.__min_epsilon = min_epsilon
        self.__decay = decay

        rd.seed(seed)

    def choose(self, q_table: Dict[str, List[float]] or np.ndarray,
               state: str,
               action_space: Dict[str, spaces.Discrete],
               available_actions: List[int]) -> int or Union[int, int]:
        """Method that computes the action choice given a Q-table, a state and an action space.
        If value is higher or equal than current epsilon, it chooses the greedy action (highest value in Q-table).
        If value is lower than current epsilon, it chooses randomly through available actions at the given state.
        The epsilon value decays at a decay rate each time this method is called until it reaches its minimum value.

        Args:
            q_table (Dict[str, List[float]] or np.ndarray): Dictionary conaining all the q values for classical
            Q-Learning or numpy array containing all non-dominated actions for Pareto Q-Learning.
            state (str): state id the agent is currently in.
            action_space (Dict[str, spaces.Discrete]): dictionary containing all action spaces for each state available.
            available_actions (List[int]): list of available indexes within that state, as not all possible actions for
            the state will be available (they depend on the link the vehicle is coming from).

        Raises:
            RuntimeError: the method raises a RuntimeError in three situations: 
                - if no available action is given.
                - if the method can't sample any of the given actions.
                - if class type of 'Q-table' is unkown.

        Returns:
            int: value (index) of the action chosen.
        """
        action = -1
        chosen_obj = -1
        self.__decay_epsilon_value()
        if self.__pickup_random:
            try:
                action = rd.choice(available_actions)
            except IndexError:
                raise RuntimeError("Couldn't take any action, no available actions given.") from IndexError
            if action not in range(action_space[state].n):
                raise RuntimeError("Available action not in state's action space.")
        elif isinstance(q_table, dict):
            action = self.__choose_dict(q_table, state, available_actions)
        elif isinstance(q_table, np.ndarray):
            action, chosen_obj = self.__choose_array(q_table, available_actions)
        else:
            raise RuntimeError("Q-table is istance of unknown class.")

        if isinstance(q_table, np.ndarray):
            return action, chosen_obj
        else:
            return action

    def __choose_dict(self, q_table: Dict[str, List[float]],
                            state: str,
                            available_actions: List[int]) -> int:
        """Method that chooses an action if the Q table given is a dictionary.

        Args:
            q_table (Dict[str, List[float]]): Dictionary conaining all the q values.
            state (str): state id the agent is currently in.
            available_actions (List[int]): list of available indexes within that state, as not all possible actions for
            the state will be available (they depend on the link the vehicle is coming from).

        Returns:
            int: value (index) of the action chosen.
        """
        max_value = max(q_table[state][action] for action in available_actions)
        equal_list = [action for action in available_actions if max_value == q_table[state][action]]

        return rd.choice(equal_list)

    def __choose_array(self, q_set: np.ndarray,
                             available_actions: List[int]) -> Union[int, int]:
        """Method that chooses an action if the Q table given is a numpy array.

        Args:
            q_set (np.ndarray): numpy array containing all non dominated actions for the current state.
            available_actions (List[int]): list of available indexes within that state, as not all possible actions for
            the state will be available (they depend on the link the vehicle is coming from).

        Raises:
            RuntimeError: the method reaises a RuntimeError if it isn't able to choose an action.

        Returns:
            int: value (index) of the action chosen.
        """
        chosen_obj = np.argmin([obj.std() for obj in q_set.T])
        max_value = max(q_set.T[chosen_obj][action] for action in available_actions)
        equal_list = [action for action in available_actions if max_value == q_set.T[chosen_obj][action]]

        try:
            action = rd.choice(equal_list)
        except IndexError:
            print("No action to choose")
            print(f"{available_actions = }")
            print(f"{max_value = }")
            print(f"{q_set.T[chosen_obj] = }")
            raise RuntimeError from IndexError
        return action, chosen_obj

    def reset(self) -> None:
        """Method that resets the current epsilon value to its initial one.
        """
        self.__epsilon = self.__initial_epsilon

    def __decay_epsilon_value(self) -> None:
        """Method that performs a decay in epsilon value if possible.
        """
        self.__epsilon = max(self.__epsilon*self.__decay, self.__min_epsilon)

    @property
    def __pickup_random(self) -> bool:
        """Property that determines if the action chosen should be optimal or random.

        Returns:
            bool: boolean value that returns True if the action chosen should be random and False otherwise.
        """
        return rd.random() < self.__epsilon
