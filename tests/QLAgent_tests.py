import unittest
from gym import spaces
from sumo_ql.exploration.epsilon_greedy import EpsilonGreedy
from sumo_ql.agent.q_learning import QLAgent


class QLAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        action_space = {
            'A0': spaces.Discrete(4),
            'A1': spaces.Discrete(3),
            'A2': spaces.Discrete(4)
        }
        action_space['A0'].seed(5)
        action_space['A1'].seed(5)

        self.__agent = QLAgent(action_space, exploration_strategy=EpsilonGreedy(0.05, 0.05, 0.99, 1))

    def test_act(self):
        self.__agent.act(state='A0', available_actions=[0, 1, 2, 3])
        self.__agent.learn(0, 'A0', 'A1', 10)

        self.__agent.act(state='A1', available_actions=[0, 1, 2])
        self.__agent.learn(1, 'A1', 'A0', 10)

        for _ in range(7):
            self.assertEqual(self.__agent.act(state='A0', available_actions=[0, 1, 2, 3]), 0)
            self.__agent.learn(0, 'A0', 'A1', 10)
            self.assertEqual(self.__agent.act(state='A1', available_actions=[0, 1, 2]), 1)
            self.__agent.learn(1, 'A1', 'A0', 10)

        self.assertEqual(self.__agent.act(state='A0', available_actions=[1, 2, 3]), 1)
        self.__agent.learn(1, 'A0', 'A1', 10)
        self.assertEqual(self.__agent.act(state='A1', available_actions=[0, 2]), 2)
