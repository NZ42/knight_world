import numpy as np
import tut_basics as tb

from plot_fxns import plotWorld, plotStateValue, plotStateActionValue, plotPolicyPi, plotGreedyPolicyQ
from rl_agents import RLAgent, RLExampleAgent
import importlib
import random

class KnightWorld(tb.GridWorld):
    def __init__(self):
        # array of possible moves
        self.actionlist = np.array(['WSW','SSW','SSE','ESE','ENE','NNE','NNW','WNW'])
        action_dict = {'WSW':0,'SSW':1,'SSE':2,'ESE':3,'ENE':4,'NNE':5,'NNW':6,'WNW':7}
        # boad shape
        shape = (5,5)
        # starting position
        start = (0,2)
        terminals = []
        # previously occupied squares
        self.obstacles = []
        # coordinates:reward of selected squares
        rewards = {(3,3):10} 
        jumps = {}
        super(KnightWorld, self).__init__(shape, start, terminals, self.obstacles, rewards, jumps)

# build and plot world
importlib.reload(tb)
world = KnightWorld()
plotWorld(world)

idx = np.arange(len(world.actionlist))[world.get_actions()]
actionlist = np.array(['WSW','SSW','SSE','ESE','ENE','NNE','NNW','WNW'])

class squire_agent(object):
    
    # remember
    # -- world.get_actions() returns an array of 8 bool values, each meaning whether the ith action is valid
    
    # -- world.stateFilter(coord) takes a 2d coordinate and returns an array containing 8 integers,
    #    each representing the square the knight would move onto if it took the ith action from state coord.
    #    (-1 means the ith action is invalid)
    
    # -- world.potMov contains 8 integers, each representing the square the knight would move onto
    #    if it took ith action (-1 if the action is invalid).
    #    It is equivalent to world.stateFilter(tb.one2twoD(agent.state)).
    
    # -- agent.valid_moves(state, actions) returns an array of actions that are both valid and taht do not move
    #    onto a previously occupied square.
    
    
    actionlist = np.array(['WSW','NNW','SSE','ESE','SSW','WNW','ENE','NNE'])
    action_dict = {'WSW':0,'NNW':1,'SSE':2,'ESE':3,'SSW':4,'WNW':5,'ENE':6,'NNE':7}
    pConv = {0:4,1:0,2:1,3:5,4:7,5:3,6:2,7:6}
    myconv = {'ENE': 7, 'ESE': 5, 'NNE': 3, 'NNW': 2, 'SSE': 1, 'SSW': 0, 'WNW': 6, 'WSW': 4}
    def __init__(self, world):
        self.world = world
        #initialize q table to zeros
        self.q = np.zeros((self.world.nstates, len(actionlist)))
    
    def display_env(self):
        # prints a grid displaying current environment
        # agent location (1), never occupied squares (0) and previously occupied squares (-1) 
        grid = np.zeros((5,5))
        grid[np.floor(self.world.get_state()/5).astype(int)][self.world.get_state()%5] = 1
        for obstacle in self.world.obstacles:
            grid[np.floor(obstacle/5).astype(int)][obstacle%5] = -1
        return grid
    
    def valid_moves(self, state):
        #returns array of valid moves
        valid_actions = [action for action in self.actionlist if self.world.potMov[self.myconv[action]] != -1]
        valid_actions = [action for action in valid_actions if self.world.potMov[self.myconv[action]] not in self.world.obstacles]
        return valid_actions
    
    def get_valid_actions(self):
        #returns array of bools (used for operations on q)
        arr = [False for i in range(8)]
        idx = [self.action_dict[move] for move in self.valid_moves(self.state)]
        for i in idx:
            arr[i] = True
        return arr

    def warnsdorffPolicy(self, state, actions):
        # replicates Warnsdorff's rule, by returning the action bringing the knight to the square with the
        # fewest onward moves.
        
        #TO DO: fix bug when the knight reaches square no. 20
        
        available_actions = self.valid_moves(state, actions)
        print(available_actions)
        pConv = {0:4,1:0,2:1,3:5,4:7,5:3,6:2,7:6}
        if len(available_actions) != 0:
            min_new_moves = 1000
            for i, action in enumerate(available_actions):
                new_cord = self.world.stateFilter(tb.oneD2twoD(state, self.world.shape))[pConv[self.world.action_dict[action]]]
                new_cord_moves = self.world.stateFilter(tb.oneD2twoD(new_cord, self.world.shape))
                new_cord_moves = [move for move in new_cord_moves if move != -1 and move not in self.world.obstacles + [state]]
                print(f"Action {action} would bring us to {tb.oneD2twoD(new_cord, self.world.shape)}, which has {len(new_cord_moves)} possible moves, that is {new_cord_moves}")
                if len(new_cord_moves) < min_new_moves:
                    min_new_moves = len(new_cord_moves)
                    best_action = action
                elif len(new_cord_moves) == min_new_moves:
                    if random.choice([True, False]):
                        min_new_moves = len(new_cord_moves)
                        best_action = action
            
            return best_action
        
        else:
            return -1
    
    def reset(self):
        # resets Knight World and agent state
        self.world.init()
        self.state = self.world.get_state()  
        
    def randPolicy(self, state, actions):
        # returns a random action bringing the knight to a never occupied square
        
        #np.random.choice() is a uniform distribution over actions
        #print(f"<randpolicy> actions: {actions}")
        #print(f"<randpolicy> actionlist: {actionlist}")
        pConv = {0:4,1:0,2:1,3:5,4:7,5:3,6:2,7:6}
        available_actions = self.actionlist[actions]
        #filAct = []
        print(f"================\n\nNEW MOVE!\n\n-@@BASE INFO\n--CURRENT SQUARE: {self.world.state}\n--POTENTIAL MOVES: {self.world.potMov}\n--OBSTACLES: {self.world.obstacles}\n\n-@@POLICY STAGE\n--AVAILABLE ACTIONS: {available_actions}")
        expAct = {}
        for eachAction in available_actions:
            #print(f"\nCURRENT STATE: {self.world.state}")
            #print(f"ACTION: {eachAction}")
            #print(f"THIS WOULD TAKE US TO: {self.world.potMov[pConv[self.world.action_dict[eachAction]]]}")
            #print(f"{self.world.obstacles}")
            if self.world.potMov[pConv[self.world.action_dict[eachAction]]] < 0:
                expAct[eachAction] = f"Was {self.world.potMov[pConv[self.world.action_dict[eachAction]]]}. Negative value!"
                available_actions = np.delete(available_actions, np.where(available_actions == eachAction))
            for x in self.world.obstacles:
                if self.world.potMov[pConv[self.world.action_dict[eachAction]]] == x:
                    expAct[eachAction] = f"Was {self.world.potMov[pConv[self.world.action_dict[eachAction]]]}. Already used!"
                    available_actions = np.delete(available_actions, np.where(available_actions == eachAction))
        for eachEntry in expAct:
            print(f"--{eachEntry} -> {expAct[eachEntry]}")
        print(f"--FILTERED ACTIONS: {available_actions}\n--BREAKDOWN: ")
        for eachAction in available_actions:
            print(f"--{eachAction}: {self.world.potMov[pConv[self.world.action_dict[eachAction]]]}")
        if len(available_actions) != 0:
            return np.random.choice(available_actions)
        else:
            self.take_action()
        
    def greedyQPolicy(self, state, actions):
        # returns action with the highest expected reward based on Q table
        idx = np.arange(8)[actions]
        return self.actionlist[idx[np.argmax(self.q[state,actions])]]

    def epsilongreedyQPolicy(self, state, actions, epsilon=0.2):
        # returns with probability epsilon a random (not greedy) action
        # and with probability 1 - epsilon the greedy action
        print("Inside epsilongreedyQPolicy")
        print(f"State = {state}")
        print(f"Actions = {actions}")
        valid_moves = self.valid_moves(state)
        print(f"Valid actions = {valid_moves}")
        print(self.display_env())
        idx = np.arange(len(world.actionlist))[actions]
        _q = self.q[state,actions]
        _max  = np.argmax(_q)
        _idx = idx[_max]
        # greedy_action = self.actionlist[idx[np.argmax(self.q[state,actions])]]
        greedy_action = self.actionlist[_idx]
        nongreedy_actions = np.delete(self.actionlist[actions],np.argwhere(self.actionlist==greedy_action))
        r = np.random.rand()
        for c in range(len(nongreedy_actions)):
            if (r<((c+1)*epsilon/len(nongreedy_actions))):
                return nongreedy_actions[c]
        return greedy_action
    
    def choose_action(self, off_pol):
        # handles action choice based on whether the agent is supposed to exploit its environment
        # (following the learned Q policy) or also explore it
        
        state = self.world.get_state()
        actions = self.get_valid_actions()
        pConv = {0:4,1:0,2:1,3:5,4:7,5:3,6:2,7:6}
        if off_pol:
            self.action = self.greedyQPolicy(state,actions)
        else:
            self.action = self.epsilongreedyQPolicy(state, actions)
        #self.action = self.warnsdorffPolicy(state, actions)
        print(f"\n-@@ACTION SELECTION\n--WE CHOSE: {self.action}, WHICH TAKES US TO {self.world.potMov[pConv[self.world.action_dict[self.action]]]}")
        #if self.action
        return self.action
    
    def take_action(self, action):
        # changes the state by taking action
        if action == -1:
            raise(ValueError("Impossible action taken"))
            return self.state, 0, True
        (self.state, self.reward, terminal) = self.world.move(action)
        #print(self.valid_moves(self.state))
        if self.valid_moves(self.state) == []:
            terminal = True
        return self.state, self.reward, terminal
        
    def evaluatePolicyQ(self, gamma, alpha, ntrials):
        # based on ntrials attempts, update the Q table with alpha as learning rate and gamma as discount factor
        delta = 1.0 # what is delta for?
        old_q = self.q
        for i in range(ntrials):
            is_terminal = False
            c = 0 
            self.reset()
            s = self.state
            #a = self.choose_action(off_pol = False)
            
            # explore from the epsilon-greedy policy
            
                
            while not is_terminal:
                c += 1
                #print(agent.display_env())
                a_prime = self.choose_action(off_pol = False) if not is_terminal else 'D'
                # below we choose a prime from the TRUE policy! 
                a_greedy = self.choose_action(off_pol = True) if not is_terminal else 'D'
                # taking an action gives terminality status.
                self.state, self.reward, is_terminal = self.take_action(a_prime) 
                
                #update 
                self.q[s, self.action_dict[a_prime]] += alpha*(self.reward + gamma * self.q[self.state, self.action_dict[a_greedy]] - self.q[s, self.action_dict[a_prime]])
                
                s = self.state
            
            #delta = np.min(np.max(np.abs(self.q - old_q)),delta)
            old_q = self.q
            
    def policyIteration(self, gamma, alpha, ntrials):
        # iterates the policy until it is stable
        print("Running TD policy iteration...")
        policyStable = False
        itr = 0
        maxiters = 1000 # catch the while loop.
        oldA = np.zeros((self.world.nstates,))
        while (not policyStable and itr < maxiters):
            self.display_env()
            itr += 1
            self.evaluatePolicyQ(gamma, alpha, ntrials)   
            policyStable = np.array_equal(oldA,np.argmax(self.q, axis=1)) # see if policy changes!
            oldA = np.argmax(self.q, axis=1)
        print("Converged after {} iterations.".format(itr))
        print(self.q)


if __name__ == "__main__":
    #create world
    importlib.reload(tb)
    world = KnightWorld()
    agent = squire_agent(world)
    state = world.get_state()
    potential_moves = world.stateFilter(tb.oneD2twoD(state, world.shape))

    #create agent
    agent = squire_agent(world)
    num_trials = 10 
    num_steps = 1
    gamma = 0.99
    alpha = 0.2
    reward_tracker = [] 
    agent.policyIteration(gamma, alpha, num_trials)

