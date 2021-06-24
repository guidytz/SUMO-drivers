import numpy as np
import random as rd
from datetime import datetime


class EpsilonGreedy:
    """Class responsible for the exploration strategy in action choice.
    """

    def __init__(self, initial_epsilon=1.0, min_epsilon=0.05, decay=0.99, seed=datetime.now()):
        """Class constructor

        Args:
            initial_epsilon (float, optional): desired starting value for epsilon. Defaults to 1.0.
            min_epsilon (float, optional): minimum value accepted for epsilon after decay. Defaults to 0.05.
            decay (float, optional): decay rate for each time it makes a choice. Defaults to 0.99.
            seed (any, optional): seed to use in random choice. Defaults to datetime.now().
        """
        self.__initial_epsilon = initial_epsilon
        self.__epsilon = initial_epsilon
        self.__min_epsilon = min_epsilon
        self.__decay = decay

        rd.seed(seed)

    def choose(self, q_table, state, action_space):
        """Method that computes the action choice given a Q-table, a state and an action space.
        If value is higher or equal than current epsilon, it chooses the greedy action (highest value in Q-table).
        If value is lower than current epsilon, it chooses randomly through possible actions at the given state.
        The epsilon value decays at a decay rate each time this method is called until it reaches its minimum value.

        Args:
            q_table (dict): dictionary containing all the q values for each action in each state available.
            state (str): state id the agent is currently in
            action_space (dict): dictionary containing all action spaces for each state available.

        Returns:
            int: value (index) of the action chosen
        """
        if rd.random() < self.__epsilon:
            action = int(action_space[state].sample())
        else:
            action = np.argmax(q_table[state])

        self.__epsilon = max(self.__epsilon*self.__decay, self.__min_epsilon)

        return action

    def reset(self):
        """Method that resets the current epsilon value to its initial one.
        """
        self.__epsilon = self.__initial_epsilon