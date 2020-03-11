'''
Created on 08/08/2014

@author: Gabriel de O. Ramos <goramos@inf.ufrgs.br>
'''
import abc

ENV_SINGLE_STATE = "***SINGLE_STATE***"

class Environment(object):
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        self._agents = {}
        
        self._episodes = 0
        self._steps = 0
        self._has_episode_ended = False
        
    #return the actions available in the specified state
    #warning: if no action is available in a given state, then return [None] 
    @abc.abstractmethod
    def get_state_actions(self, state):
        return
    
    #run an entire episode
    @abc.abstractmethod
    def run_episode(self, max_steps=-1):
        pass
    
    #run a single step within an episode
    @abc.abstractmethod
    def run_step(self):
        pass
    
    #check if the episode has ended
    @abc.abstractmethod
    def has_episode_ended(self):
        return
    
    #reset all attributes
    def reset_all(self):
        self._episodes = 0
        
        for l in self._agents.values():
            l.reset_all()
        
    #reset all episode-related attributes
    def reset_episode(self):
        self._steps = 0
        self._has_episode_ended = False
        
        for l in self._agents.values():
            l.reset_episodic(self._episodes)
    
    #register a agent within the environment
    def register_agents(self, agents):
        for agent in agents:
            if agent.get_name() in self._agents:
                raise Exception("agent %s has already registered to %s!" % (agent.get_name(), self.__str__()))
            self._agents[agent.get_name()] = agent
    
    #defines what must happen when one prints the agent
    def __str__(self):
        return "%s %s" % (self.__class__.__name__, self.__class__.__bases__[0].__name__)

        
