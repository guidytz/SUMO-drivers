'''
Created on 06/08/2014

@author: Gabriel de O. Ramos <goramos@inf.ufrgs.br>
'''
import abc

class Learner(object):
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, name, env, child_instance):
        
        self._name = name
        
        self._env = env
        
        self._state = None
        self._action = None
        
        self._episode = None
        
        # subset of available actions in the current state
        # (used only when NOT ALL actions of the current state are available)
        # PS: when all actions are available, the value must be set to None 
        #self._available_actions = None
        
    # reset all attributes
    @abc.abstractmethod
    def reset_all(self):
        pass
        
    # reset all episode-related attributes
    def reset_episodic(self, episode):
        self._episode = episode
    
    # return the agent's current state and the action to be performed by him. 
    # the state may be passed as parameter if the reasoning is being made in advance (SUMO env, eg).
    # the method is divided into three parts to ensure a sort of "milestone" among them (eg, 
    # if one needs to ensure that all agents have finished a first piece of code before a second
    # piece is started; this feature was defined mainly to allow agents to communicate among 
    # themselves before making a final decision)   
    @abc.abstractmethod
    def take_action(self, state=None, available_actions=None):
        return
        
    # send to the agent the feedback corresponding to his action.
    # the method is divided into three parts following the same reasoning of method act.
    @abc.abstractmethod
    def process_feedback(self, reward, new_state, prev_state=None, prev_action=None):
        pass
    
    # return the name of the learner
    def get_name(self):
        return self._name
    
    # return the current state of the learner within the environment
    def get_state(self):
        return self._state
    
    # defines what must happen when one prints the learner
    def __str__(self):
        return "%s %s ()" % (self.__class__.__name__, self._name)
