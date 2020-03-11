'''
Created on 04/08/2014

@author: Gabriel de O. Ramos <goramos@inf.ufrgs.br>
'''
from exploration import ExplorationStrategy
from tools import sampling
import random

class EpsilonGreedy(ExplorationStrategy):
    
    def __init__(self, epsilon=1, min_epsilon=0.0, decay_rate=0.999):#to avoid decay rate, set it to 0.0 
        self._epsilon_ini = epsilon
        self._epsilon = epsilon
        self._min_epsilon = min_epsilon
        self._decay_rate = decay_rate
        self._last_episode = 0
        
    #Return an action, given an actions dictionary in the form action:Q-value
    def choose(self, action_dict, episode):
		
        #update epsilon value
        if  self._last_episode != episode and self._decay_rate > 0.0 and self._epsilon > self._min_epsilon:
            self._epsilon = self._epsilon * self._decay_rate
        self._last_episode = episode
        
        #select an action
        r = random.random()
        i = -1
        if r < self._epsilon:
            #select an action uniformly at random (exploration)
            i = random.randint(0, len(action_dict)-1)
            #~ print 'exploration', self._epsilon
        else:
            #select an action greedily (exploitation)
            i = sampling.reservoir_sampling(action_dict.values(), True)
            #~ print 'exploitation', self._epsilon
        
        #update epsilon value
        if self._decay_rate > 0.0:
            self._epsilon = self._epsilon * self._decay_rate
        
        #return selected action
        return list(action_dict.keys())[i]
        
    def update_epsilon_manually(self, val):
        self._epsilon = val
    # def update_epsilon_manually(self):
    # def update_epsilon_manually(self):
    #     if self._manual_decay == False:
    #         print ('[WARNING] Manual decay should not be used; Epsilon was set with automatic decay!')
	# 	# if self._manual_decay == False:
	# 	# 	print ('[WARNING] Manual decay should not be used; Epsilon was set with automatic decay!')
	# 	if self._epsilon > self._min_epsilon:
	# 		if self._decay_rate > 0.0:
	# 			self._epsilon = self._epsilon * self._decay_rate        
        
    def reset(self):
        #epsilon greedy is not episodic, so nothing to do here
        self._epsilon = self._epsilon_ini
    
    def reset_all(self):
        pass
