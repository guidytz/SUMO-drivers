from typing import Dict
from typing import List
from gym import spaces

from sumo_ql.exploration.epsilon_greedy import EpsilonGreedy

class QLAgent:
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
        self.__action_space = action_space
        self.__alpha = alpha
        self.__gamma = gamma
        self.__exploration_strategy = exploration_strategy
        self.__q_table = {state: [0 for _ in range(self.__action_space[state].n)]
                          for state in self.__action_space.keys()}

    def act(self, state: str, available_actions: List[int]) -> int:
        """Method that performs, i.e. chooses, a new action for the agent.

        Args:
            state (str): the state the agent is currently in.
            available_actions (List[int]): A list containing the available actions for the given state at the moment.

        Returns:
            int: the action chosen in the agent's current state.
        """
        return self.__exploration_strategy.choose(self.__q_table, state, self.__action_space, available_actions)

    def learn(self, action: int, current_state: str, next_state: str, reward: int) -> None:
        """Method that updates the Q-table based on the reward received for the action taken at the given current state,


        Args:
            action (int): The action taken.
            current_state (str): The state where the action was taken.
            next_state (str): The next state reached with the action.
            reward (int): The reward received for taking the action within the given state.

        Raises:
            RuntimeError: the method raises a RuntimeError if the new state given is not in Q-table
        """
        if next_state not in self.__q_table:
            raise RuntimeError("Invalid state: not found in Q-table!")

        max_future_value = max(self.__q_table[next_state])
        self.__q_table[current_state][action] += self.__alpha * (reward + self.__gamma * max_future_value -
                                                                 self.__q_table[current_state][action])
