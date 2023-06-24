import unittest

from gym import spaces

from sumo_drivers.exploration.epsilon_greedy import EpsilonGreedy


class EpsilonGreedyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.__action_space = {
            'A0': spaces.Discrete(4),
            'A1': spaces.Discrete(3)
        }
        self.__q_table = {'A0': [2, 7, 3, 6], 'A1': [1, 2, 3]}

        # values 2 and 3 are the first values to show up with this seed
        self.__action_space['A0'].seed(5)

    def test_choose(self):
        # 10 and 14 call for random < 0.05 with this seed
        exploration = EpsilonGreedy(0.05, 0.05, 0.99, 1)

        for _ in range(9):
            action = exploration.choose(self.__q_table, 'A0', self.__action_space, available_actions=[1, 2, 3])
            self.assertEqual(action, 1)

        action = exploration.choose(self.__q_table, 'A0', self.__action_space, available_actions=[0, 2, 3])
        self.assertEqual(action, 2)

        for _ in range(3):
            action = exploration.choose(self.__q_table, 'A0', self.__action_space, available_actions=[0, 1, 2])
            self.assertEqual(action, 1)

        action = exploration.choose(self.__q_table, 'A0', self.__action_space, available_actions=[0, 1, 2])
        self.assertEqual(action, 2)

    def test_reset(self):
        exploration = EpsilonGreedy(1.0, 0.25, 0.5, 1)
        available_actions = [0, 1, 2, 3]
        action = exploration.choose(self.__q_table, 'A0', self.__action_space, available_actions)
        self.assertEqual(action, 2)

        action = exploration.choose(self.__q_table, 'A0', self.__action_space, available_actions)
        self.assertEqual(action, 1)

        exploration.reset()

        action = exploration.choose(self.__q_table, 'A0', self.__action_space, available_actions)
        self.assertEqual(action, 3)
