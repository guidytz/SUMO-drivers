'''
Created on 04/08/2014

@author: Gabriel de O. Ramos <goramos@inf.ufrgs.br>
'''
import abc

class ExplorationStrategy:
    
    __metaclass__ = abc.ABCMeta
    
    #Return an action, given an actions dictionary in the form action:Q-value
    @abc.abstractmethod
    def choose(self, action_dict, episode):
        return
    
    #Called only in the beginning of each episode
    @abc.abstractmethod
    def reset(self):
        pass
    
    #Called in the beginning of the simulation
    @abc.abstractmethod
    def reset_all(self):
        pass
