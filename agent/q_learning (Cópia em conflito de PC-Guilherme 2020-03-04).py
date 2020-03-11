'''
Created on 04/08/2014

@author: Gabriel de O. Ramos <goramos@inf.ufrgs.br>
'''
from agent import Learner

class QLearner(Learner):
    
    def __init__(self, name, env, starting_state, goal_state, alpha, gamma, exp_strategy):
        
        super(QLearner, self).__init__(name, env, self)

        self._starting_state = starting_state
        self._goal_state = goal_state
        
        self._exp_strategy = exp_strategy
        
        self._alpha = alpha
        self._gamma = gamma
        
        self._initialise_Q_table()
        
        self.reset_episodic(0)
    
    #initialize the Q-table.
    #in the beginning, only the entries corresponding to initial state
    #are populated. The other entries are populated on the fly.
    def _initialise_Q_table(self):#TODO - replace by __check_and_create_Q_table_entry
        self._QTable = {}
        
        # for state in states_IDs:
            # self._QTable[state] = dict({a:0 for a in self._env.get_state_actions(state)})
        self._QTable[self._starting_state] = dict({a:0 for a in self._env.get_state_actions(self._starting_state)})
    
    def reset_episodic(self, episode):
        super(QLearner, self).reset_episodic(episode)
		
        self._state = self._starting_state
        self._action = None
        self._accumulated_reward = 0.0
        
        self._exp_strategy.reset()
        
        self._has_arrived = False
    
    def take_action(self, state=None, available_actions=None):
        
        #the state may be passed as parameter if the reasoning is not being made
        #regarding the current state (as is the case in SUMO env, eg)
        if state == None:
            state = self._state
        else:
            self.__check_and_create_Q_table_entry(state)
        
        #if not all actions are available, select the subset and corresponding Q-values
        available = self._QTable[state]
        if available_actions != None:#TODO
            available = {}
            for a in available_actions:
                available[a] = self._QTable[state][a]
        #print state
        #print 'available: %s'%available
        #print 'all: %s'%self._QTable[state]
        
        if not available:
            self._has_arrived = True
        else:
            #choose action according to the the exploration strategy
            self._action = self._exp_strategy.choose(available, self._episode)
        
        #print "Action %s taken in state %s" % (self._action, self._state)
        
        #return action to take
        return [state, self._action]
    
    #check whether the given state is already in the Q-table, if not, create it
    #PS: as the Q-table is created on-the-fly, some states may not be in the table yet
    def __check_and_create_Q_table_entry(self, state):
        try:
            self._QTable[state].keys()
        except:
            self._QTable[state] = dict({a:0 for a in self._env.get_state_actions(state)})

    def switch_epsilon(self, new_val):
        self._exp_strategy.update_epsilon_manually(new_val)
    
    def get_Q_table(self):
        return self._QTable.copy()
    
    def process_feedback(self, reward, new_state, prev_state=None, prev_action=None):
        
        #print reward
        #print new_state
        #print self._state
        #print self._action
        
        state = prev_state
        if state == None:
            state = self._state
        
        action = prev_action
        if action == None:
            action = self._action
        
        #print "After performing action %s in state %s, the new state is %s and the reward %f" % (action, state, new_state, reward)
        
        #check whether new_state is already in Q-table
        self.__check_and_create_Q_table_entry(state)
        self.__check_and_create_Q_table_entry(new_state)
        
        #update Q table with cur_state and action
        try:
            maxfuture = 0.0
            if self._QTable[new_state]: #dictionary not empty
                maxfuture = max(self._QTable[new_state].values())
            
            self._QTable[state][action] += self._alpha * (reward + self._gamma * maxfuture - self._QTable[state][action])
        except Exception:
            print ("Nooooooooooooooooooooooooo!!!!")
            #print new_state
            #print self._env.get_state_actions(new_state)
            #print self._QTable
            #print str(e)
        
        #update curr_state = new_state
        self._state = new_state
        
        #update the subset of actions that are available on the new state (None if all are available) 
        #self._available_actions = available_actions
        
        #update accumulated reward
        self._accumulated_reward += reward
        
        #check whether an ending state has been reached
        if new_state == self._goal_state or not list(self._QTable[new_state].keys())[0]:
            self._has_arrived = True
        
    def has_arrived(self):
        return self._has_arrived