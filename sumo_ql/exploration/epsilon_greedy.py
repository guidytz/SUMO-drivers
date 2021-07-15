import random as rd
from datetime import datetime
from typing import Dict
from typing import List
from gym import spaces

MAX_SAMPLE_COUNTER = 30


class EpsilonGreedy:
    """Class responsible for the exploration strategy in action choice.

    Args:
        initial_epsilon (float, optional): desired starting value for epsilon. Defaults to 1.0.
        min_epsilon (float, optional): minimum value accepted for epsilon after decay. Defaults to 0.05.
        decay (float, optional): decay rate for each time it makes a choice. Defaults to 0.99.
        seed (int, optional): seed to use in random choice. Defaults to datetime.now().
    """

    def __init__(self,
                 initial_epsilon: float = 1.0,
                 min_epsilon: float = 0.05,
                 decay: float = 0.99,
                 seed: int = datetime.now()) -> None:
        self.__initial_epsilon = initial_epsilon
        self.__epsilon = initial_epsilon
        self.__min_epsilon = min_epsilon
        self.__decay = decay

        rd.seed(seed)

    def choose(self, q_table: Dict[str, List[float]],
               state: str,
               action_space: Dict[str, spaces.Discrete],
               available_actions: List[int]) -> int:
        """Method that computes the action choice given a Q-table, a state and an action space.
        If value is higher or equal than current epsilon, it chooses the greedy action (highest value in Q-table).
        If value is lower than current epsilon, it chooses randomly through available actions at the given state.
        The epsilon value decays at a decay rate each time this method is called until it reaches its minimum value.

        Args:
            q_table (Dict[str, List[float]]): dictionary containing all the q values for each action in each state
            available.
            state (str): state id the agent is currently in
            action_space (Dict[str, spaces.Discrete]): dictionary containing all action spaces for each state available.
            available_actions (List[int]): list of available indexes within that state, as not all possible actions for
            the state will be available (they depend on the link the vehicle is coming from).

        Raises:
            RuntimeError: if no available action is given.
            RuntimeError: if the method can't sample any of the given actions.

        Returns:
            int: value (index) of the action chosen
        """
        if rd.random() < self.__epsilon:
            counter = 0
            while counter < MAX_SAMPLE_COUNTER:
                action = int(action_space[state].sample())
                if action in available_actions:
                    break
                counter += 1
            if counter >= MAX_SAMPLE_COUNTER:
                if len(available_actions) == 1:
                    action = available_actions[0]
                elif len(available_actions) == 0:
                    raise RuntimeError("Couldn't take any action, no available actions given.")
                else:
                    print(f"{available_actions = }")
                    print(f"{state = }")
                    print(f"{action = }")
                    print(f"{action_space[state] = }")
                    raise RuntimeError("Something went wrong in sample, couldn't take any of the available actions.")
        else:
            available_values = []
            for action in range(len(q_table[state])):
                if action in available_actions:
                    available_values.append(q_table[state][action])
            max_value = max(available_values)
            equal_list = list()
            for index, value in enumerate(q_table[state]):
                if value == max_value and index in available_actions:
                    equal_list.append(index)

            if len(equal_list) != 1:
                action = rd.choice(equal_list)
            else:
                action = equal_list[0]

        self.__decay_epsilon_value()

        return action

    def reset(self) -> None:
        """Method that resets the current epsilon value to its initial one.
        """
        self.__epsilon = self.__initial_epsilon

    def __decay_epsilon_value(self) -> None:
        """Method that performs a decay in epsilon value if possible.
        """
        self.__epsilon = max(self.__epsilon*self.__decay, self.__min_epsilon)
