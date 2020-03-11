'''
Created on 08/08/2014

@author: Gabriel de O. Ramos <goramos@inf.ufrgs.br>
'''
from exploration import ExplorationStrategy
from tools import sampling
import math

class Boltzmann(ExplorationStrategy):
    
    def __init__(self, temperature=0.1):
        self._temperature = temperature
        
    #Return an action, given an actions dictionary in the form action:Q-value
    def choose(self, action_dict):
        
        #calculate the probability of each action
        values = []
        sumup = 0
        for av in action_dict.values():
            v = (math.e ** av) / self._temperature
            sumup += v
            values.append(v)
        
        #sample one of the actions (index is returned)
        i = sampling.probability_vector(values, sumup)
        
        #return selected action
        return action_dict.keys()[i]
    
    def reset_episodic(self):
        #Boltzmann is not episodic, so nothing to do here
        pass
    
    def reset_all(self):
        #Boltzmann does not vary its parameters once defined
        pass