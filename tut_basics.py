import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import cm
import matplotlib.colorbar as colorbar
import matplotlib.colors as colors

## --------------------------------------------------------------------------------
## MULTIARMED BANDIT
## --------------------------------------------------------------------------------
def bandit1(): 
    return np.random.choice([0,1]) 

def bandit2():
    return np.random.choice([1,10], p=[0.7,0.3])

def bandit3():
    return np.random.choice([1,10],p=[0.9,0.1])

## --------------------------------------------------------------------------------
## GRIDWORLD 
## --------------------------------------------------------------------------------
def run_episode(agent, print_details=False):
    print("Running episode...")
    is_terminal = False
    agent.reset()
    c = 0  # step counter
    while (is_terminal == False):
        c += 1 # increase step counter
        prev_state = oneD2twoD(agent.state,agent.world.shape)
        action = agent.choose_action()
        is_terminal = agent.take_action(action)
        state = oneD2twoD(agent.state,agent.world.shape)
        if print_details:
            print("Step %d: move from (%d,%d) to (%d,%d), reward = %.2f" % (c,prev_state[0],prev_state[1],state[0],state[1],agent.reward))
    print(f"Terminated after {c} episodes.")


def oneD2twoD(idx, shape):    
	return (int(idx / shape[1]), np.mod(idx,shape[1]))

def twoD2oneD(r, c, shape):
	return r * shape[1] + c

class GridWorld(object):
    action_dict = {'WSW':0,'SSW':1,'SSE':2,'ESE':3,'ENE':4,'NNE':5,'NNW':6,'WNW':7}
    def __init__(self, shape, start, terminals, obstacles, rewards, jumps):
        self.action_dict = {'WSW':0,'SSW':1,'SSE':2,'ESE':3,'ENE':4,'NNE':5,'NNW':6,'WNW':7}
        self.shape = shape
        self.nstates = shape[0]*shape[1]
        self.start = twoD2oneD(start[0], start[1], shape)
        self.state = self.start
        if isinstance(terminals, tuple):
            self.terminal2D = [terminals]
            self.terminal = [twoD2oneD(terminals[0],terminals[1],shape)]
        else:
            self.terminal2D = terminals
            self.terminal = [twoD2oneD(r,c,shape) for r,c in terminals]
        if isinstance(obstacles, tuple):
            self.obstacle2D = [obstacles]
            self.obstacle = [twoD2oneD(obstacles[0],obstacles[1],shape)]
        else:
            self.obstacle2D = obstacles
            self.obstacle = [twoD2oneD(r,c,shape) for r,c in obstacles]
        self.jump = jumps
        self.jump_from = [twoD2oneD(x,y,shape) for x,y in list(jumps.keys())]
        self.jump_to = [twoD2oneD(x,y,shape) for x,y in list(jumps.values())]
        self.stateFilter(oneD2twoD(self.state,self.shape))
        self.buildTransitionMatrices()

        self.R = np.zeros((self.nstates,))  # rewards received upon leaving state
        self.stateRange()
        
        for r,c in list(self.rewards.keys()):
            self.R[twoD2oneD(r,c,self.shape)] = self.rewards[(r,c)]

    def stateRange(self):
        self.rewards = {}
        for eachX in range(self.shape[0]):
            for eachY in range(self.shape[0]):
                if (eachX,eachY) not in self.rewards.keys():
                    self.rewards[(eachX,eachY)] = 1

    def stateFilter(self,coord):
        # pass in coordinate
        # calculate all possible moves
        x_i,y_i = coord
        #print(f"\n<statefilter> CALCULATING MOVE SETS...\n<statefilter> INITIAL COORDINATES: {coord}")
        unfilMov = [] #All possible moves.
        self.potMov = [] #All moves that fit on the board.

        #Possible coordinate values.
        potX = [x_i+2,x_i-2,x_i+1,x_i-1]
        potY = [y_i-2,y_i+2,y_i-1,y_i+1]

        #Possible combinations.
        unfilMov = [[potX[0],potY[2]] , [potX[0],potY[3]], [potX[1],potY[2]] , [potX[1],potY[3]], [potX[2],potY[0]], [potX[2],potY[1]], [potX[3],potY[0]], [potX[3],potY[1]]]

        #Validating combinations according to the board dimensions
        for eachSet in unfilMov:
            #print(f"<statefilter> Each Set iteration: {eachSet}")
            if (eachSet[0] in np.arange(self.shape[0])) and (eachSet[1] in np.arange(self.shape[1])):
                self.potMov.append(twoD2oneD(eachSet[0], eachSet[1], self.shape))
            else:
                self.potMov.append(-1)

        #return potMoves
        #self.potMov = twoD2oneD(self.potMov[1],self.potMov[1], self.shape)    
        return self.potMov   
            
    def termTest(self,newState):
        self.stateFilter(oneD2twoD(newState,self.shape))
        stateSpace = len(self.potMov)
        print(f"-@@TERMINAL TESTING\n--Potential Moves:{self.potMov}")
        for x in self.potMov:
            if x < 0 or (x in self.obstacles):
                print(f"--{x} isn't valid!")
                stateSpace -= 1
        print(f"--That means there are {stateSpace} viable moves.")
        if stateSpace == 0:
            print("--TRIAL IS TERMINAL!\n")
            return(True)
        else:
            print("--On we go...\n")
            return(False)
        
    def nanCheck(self,Spot,potEntry):
        #if self.potMov[potEntry] == np.nan:  
        if potEntry == np.nan:
            self.P[int(Spot),int(self.state),int(self.potMov[potEntry])] = 0
        else:
            self.P[int(Spot),int(self.state),int(self.potMov[potEntry])] = 1
        
    def buildTransitionMatrices(self):
        # initialize
        self.P = np.zeros((8, self.nstates, self.nstates))
        
        self.stateFilter(oneD2twoD(self.state,self.shape))
        
        self.nanCheck(0,4)
        self.nanCheck(1,0)
        self.nanCheck(2,1)
        self.nanCheck(3,5)
        self.nanCheck(4,7)
        self.nanCheck(5,3)
        self.nanCheck(6,2)
        self.nanCheck(7,6)
        
        #self.P[0,self.state,self.potMov[4]] = 1
        #self.P[1,self.state,self.potMov[0]] = 1
        #self.P[2,self.state,self.potMov[1]] = 1
        #self.P[3,self.state,self.potMov[5]] = 1
        #self.P[4,self.state,self.potMov[7]] = 1
        #self.P[5,self.state,self.potMov[3]] = 1
        #self.P[6,self.state,self.potMov[2]] = 1
        #self.P[7,self.state,self.potMov[6]] = 1
        
        # remove select states
        endlines = list(range(self.shape[1]-1,self.nstates-self.shape[1],self.shape[1]))
        endlines2 = [x+1 for x in endlines]
        endlines3 = list(range(self.shape[1]-1))
        endlines4 = []
        
        for i in range(self.shape[1]-1):
            endlines4.append(self.shape[1]**2-1 - i)
        
        for i in range(8):
            self.P[i,endlines,endlines2] = 0
            self.P[i,endlines3,endlines4] = 0        
            
        for i in range(4):
            self.P[i,:,self.obstacles] = 0  	# remove transitions into obstacles
            self.P[i,self.obstacles,:] = 0  	# remove transitions from obstacles
            self.P[i,self.terminal,:] = 0  	# remove transitions from terminal states
            self.P[i,self.jump_from,:] = 0 	# remove neighbor transitions from jump states 
            
        print(f"-MATRIX BUILT!\n")
    
    def init(self):
        self.state = self.start
        self.obstacles = []
        self.buildTransitionMatrices()

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def get_actions(self):
        return np.any(self.P[:,self.state,:],axis=1)

    def move(self, action):  
        #print(f"<move1> self.P[self.action_dict[action],self.state,:]: {self.P[self.action_dict[action],self.state,:]}")
        #print(f"<move> SELF.STATE @ PRE-MOVE: {self.state}")
        #print(f"<move> oneD2twoD(self.state,self.shape): {oneD2twoD(self.state,self.shape)}")

        if not self.get_actions()[self.action_dict[action]]:
            print(f"-<move> ACTION: {action}")
            print(f"-<move> POTMOVES: {self.potMov}")
            print(f"-<move> self.get_actions(): {self.get_actions()}")
            print(f"-<move> self.action_dict(): {self.action_dict}")
            print(f"-<move> self.get_actions()[self.action_dict[action]]: {self.get_actions()[self.action_dict[action]]}")
            
       
        #print(f"<move2> self.P[self.action_dict[action],self.state,:]: {self.P[self.action_dict[action],self.state,:]}")
        #print(f"self.rewards: {self.rewards}")
        reward = self.R[self.state]
        
        #print(f"<move3> self.P[self.action_dict[action],self.state,:]: {self.P[self.action_dict[action],self.state,:]}")
        self.obstacles.append(self.state)
        
        #print(f"<move4> self.P[self.action_dict[action],self.state,:]: {self.P[self.action_dict[action],self.state,:]}")
        #self.P[:,:,self.state] = 0  
        
        #print(f"<move> action: {action}\n<move> self.action_dict: {self.action_dict}")
        #print(f"<move> self.P[self.action_dict[action],self.state,:]: {self.P[self.action_dict[action],self.state,:]}")
        #print(f"<move6> np.nonzero(self.P[self.action_dict[action],self.state,:]):{np.nonzero(self.P[self.action_dict[action],self.state,:])}")
        #print(f"<move> np.nonzero(self.P[self.action_dict[action],self.state,:])[0]:{np.nonzero(self.P[self.action_dict[action],self.state,:])[0]}")
        #print(f"<move> np.nonzero(self.P[self.action_dict[action],self.state,:])[0][0]:{np.nonzero(self.P[self.action_dict[action],self.state,:])[0][0]}")
        
        print(f"\n-@@MOVE FUNCTION")
        #print(f"--PICKING {action} TO MOVE FROM {self.state} TO {np.nonzero(self.P[self.action_dict[action],self.state,:])[0][0]}.\n")
        #print(f"--OBSTACLES: {self.obstacles}\n")
        self.state = np.nonzero(self.P[self.action_dict[action],self.state,:])[0][0]  # update to new state
        #print(f"<move> SELF.STATE @ 4: {self.state}")
        
        self.stateFilter(oneD2twoD(self.state,self.shape))
        self.nanCheck(0,4)
        self.nanCheck(1,0)
        self.nanCheck(2,1)
        self.nanCheck(3,5)
        self.nanCheck(4,7)
        self.nanCheck(5,3)
        self.nanCheck(6,2)
        self.nanCheck(7,6)
        
        self.buildTransitionMatrices()
        
        is_terminal = self.termTest(self.state)
        
        # check if this is a terminal state
        #if len(self.available_actions) == 0:
        #    print(f"NO MORE MOVES AVAILABLE!")
        #    is_terminal = True
        #else:
        #    is_terminal = False
        
        print(f"MOVE FINISHED! ONWARDS!\n\n================")
        
        return (self.state, reward, is_terminal)

class CliffWorld(GridWorld):
    def __init__(self):
        shape = (3,7)
        start = (2,0)
        terminals = (2,6)
        obstacles = []
        rewards = {}
        jumps = {}
        for i in range(14):
            rewards[oneD2twoD(i,shape)] = -1
        rewards[(2,0)]=-1
        for i in range(15,21):
            rewards[oneD2twoD(i,shape)] = -10
            jumps[oneD2twoD(i,shape)] = (2,0)
        super(CliffWorld, self).__init__(shape, start, terminals, obstacles, rewards, jumps)

class env(object):
	def __init__(self, shape, start, terminals, obstacles, jumps, actions, rewards, **kwargs):
		"""
		Args:
			shape (tuple): defines the shape of the gridworld
			start (tuple): defines the starting position of the agent
			terminals (tuple or list): defines terminal states (end of episodes)
			obstacles (tuple or list): defines obstacle squares (cannot be visited)
			rewards (dictionary): states to reward values for EXITING those states
			jumps (dictionary): non-neighbor state to state transitions
		"""
		self.shape = shape
		self.rows = shape[0]
		self.cols = shape[1]
		self.strt = start
		self.term = terminals
		self.obst = obstacles
		self.jump = jumps
		self.acts = actions
		self.rwds = rewards

		self.free = []
		for i in range(self.rows):
		    for j in range(self.cols):
		        if (i,j) in self.obst:
		            pass
		        else:
		            self.free.append((i,j))
		self.state = self.strt

		self.nstates = self.rows*self.cols
		self.jump_from = [twoD2oneD(x,y,shape) for x,y in list(jumps.keys())]
		self.jump_to = [twoD2oneD(x,y,shape) for x,y in list(jumps.values())]

		self.R = np.zeros((self.nstates,))  # rewards received upon leaving state
		for r,c in list(rewards.keys()):
			self.R[twoD2oneD(r,c,self.shape)] = rewards[(r,c)]

	def move(self, state, action):
		x = state[0]
		y = state[1]
		if action not in self.acts: 
			print('Invalid action selection')
		if action == 'down':
			x = x
			if y <self.rows-1:
				y = y + 1
			else:
				y = y
		if action == 'up':
			x = x
			if y > 0:
				y = y - 1
			else:
				y = y
		if action == 'right':
			y = y
			if x<self.cols-1:
				x = x+1
			else:
				x = x
		if action == 'left':
			y = y
			if x > 0:
				x = x-1

		if state in self.jump and action == 'jump':
			x = self.jump[state][0]
			y = self.jump[state][1]

		self.state = (x,y)
		return self.state

	def reward(self, state):
		if state in self.rwds:
			return self.rwds[state]
		else:
			return 0 

	def transition_matrix(self):
		# initialize empty transition matrix 
		self.P = np.zeros((len(self.acts), self.nstates, self.nstates))
		for ind, action in enumerate(self.acts[:-1]):
			for cur_state in self.free:
				next_state = self.move(cur_state, action)
				state_from_ind = twoD2oneD(cur_state[0],cur_state[1],self.shape)
				state_to_ind = twoD2oneD(next_state[0],cur_state[1],self.shape)

				self.P[ind, state_from_ind, state_to_ind] = 1



## plot environment for gridworld problem 
def plot_grid(env, **kwargs):
    # optional argument to save figure you create
    save = kwargs.get('save', False)
    
    # make figure object
    fig = plt.figure()
    ax = fig.gca()
    cmap = cm.get_cmap('Set1')

    # get number of rows and columns from the shape of the grid (passed to function)
    n_rows = env.rows
    n_cols = env.cols

    # get info about other aspects of environment
    obst = env.obst
    start_loc = env.strt
    terminal_states = env.term
    jumps = env.jump

    # make grid environment
    grid = np.zeros((n_rows, n_cols))
    # populate environments with obstacles
    for i in obst: 
        grid[i] = 1 

    # show the basic grid    
    plt.imshow(grid, cmap ='bone_r')
    # add gridlines for visibility
    plt.vlines(np.arange(n_cols)+.5, ymin=0-.5, ymax = n_rows-.5, color='k', alpha=0.2)
    plt.hlines(np.arange(n_rows)+.5, xmin=0-.5, xmax = n_cols-.5, color='k', alpha=0.2)
    
    # add annotation for start location
    plt.annotate('S', (start_loc[1], start_loc[0]))
    
    # add annotations for terminal location/s
    for i in terminal_states:
        ax.add_patch(patches.Rectangle(np.add((i[1],i[0]),(-.5,-.5)), 1,1, lw = 3, ec = 'k', color = 'gray', alpha=0.5))
        plt.annotate('T', (i[1],i[0]))

    # add annotations for jump states
    for ind, i in enumerate(jumps.items()):
        start_jump = i[0]
        end_jump = i[1]
        colour = cmap(ind)        
        ax.add_patch(patches.Rectangle(np.add((start_jump[1],start_jump[0]),(-.5,-.5)), 1,1, fill=False, ec = colour, lw = 2, ls = "--"))
        ax.add_patch(patches.Rectangle(np.add((end_jump[1],end_jump[0]),(-.5,-.5)), 1,1, color = colour ,alpha=0.5))

    # statement for saving if optional arg save==True
    if save: 
        plt.savefig('./gridworld.png', format='png')

# value plots for gridworld env
def plot_valmap(env, value_array, save=False, **kwargs):
	'''
	:param maze: the environment object
	:param value_array: array of state values
	:param save: bool. save figure in current directory
	:return: None
	'''
	title = kwargs.get('title', 'State Value Estimates')
	vals = value_array.copy()

	fig = plt.figure()
	ax1 = fig.add_axes([0, 0, 0.85, 0.85])
	axc = fig.add_axes([0.75, 0, 0.05, 0.85])
	vmin, vmax = kwargs.get('v_range', [np.min(value_array), np.max(value_array)])

	cmap = plt.cm.Spectral_r
	cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
	for i in env.obst:
		vals[i[0], i[1]] = np.nan
	
	cb1 = colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
	ax1.imshow(vals, cmap=cmap, vmin = vmin, vmax = vmax)

	ax1.set_aspect('equal')
	plt.title(title)
	plt.show()


# policy plots for gridworld env
def make_arrows(action, probability):
	'''
	:param action:
	:param probability:
	:return:
	'''
	if probability == 0:
		dx, dy = 0, 0
		head_w, head_l = 0, 0
	else:
		if action == 0:  # N
			dx = 0
			dy = -.25
		elif action == 1:  # E
			dx = .25
			dy = 0
		elif action == 2:  # W
			dx = -.25
			dy = 0
		elif action == 3:  # S
			dx = 0
			dy = .25
		elif action == 4:  # stay
			dx = -.1
			dy = -.1
		elif action == 5:  # poke
			dx = .1
			dy = .1

		head_w, head_l = 0.1, 0.1

	return dx, dy, head_w, head_l

