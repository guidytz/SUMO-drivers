import unittest
import unittest.mock as mock
from sumo_ql import exploration
from sumo_ql.exploration import EpsilonGreedy
from gym import spaces

class EpsilonGreedyTest(unittest.TestCase):

    def setUp(self) -> None:
        self.__action_space = {'A0': spaces.Discrete(4), 'A1': spaces.Discrete(3)}
        self.__q_table = {'A0': [2, 7, 3, 6], 'A1': [1, 2, 3]}
        self.__action_space['A0'].seed(5) # values 2 and 3 are the first values to show up with this seed
        

    def test_choose(self):
        exploration = EpsilonGreedy(0.05, 0.05, 0.99, 1) # 10 and 14 call for random < 0.05 with this seed

        for _ in range(9):
            action = exploration.choose(self.__q_table, 'A0', self.__action_space)
            self.assertEqual(action, 1)

        action = exploration.choose(self.__q_table, 'A0', self.__action_space)
        self.assertEqual(action, 2)

        for _ in range(3):
            action = exploration.choose(self.__q_table, 'A0', self.__action_space)
            self.assertEqual(action, 1)
        
        action = exploration.choose(self.__q_table, 'A0', self.__action_space)
        self.assertEqual(action, 3)

    def test_reset(self):
        exploration = EpsilonGreedy(1.0, 0.25, 0.5, 1)

        action = exploration.choose(self.__q_table, 'A0', self.__action_space)
        self.assertEqual(action, 2)

        action = exploration.choose(self.__q_table, 'A0', self.__action_space)
        self.assertEqual(action, 1)

        exploration.reset()

        action = exploration.choose(self.__q_table, 'A0', self.__action_space)
        self.assertEqual(action, 3)
