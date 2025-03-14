import numpy as np

def load_chain(file, gamma):
    
	ProbMatrix = np.load(file)
    
	n_states = ProbMatrix.shape[1]
    
	combined_matrix = (1 - gamma) * ProbMatrix[0] + gamma * ProbMatrix[1]
    
	state_space = tuple(str(i) for i in range(n_states))
    
	return (state_space, combined_matrix)

Lm = 10

def observ2state(observ):
    dimsizes = [Lm, Lm]
    return np.ravel_multi_index(observ, dimsizes)

# Auxiliary function to convert state index to state representation
def state2observ(state):
    dimsizes = [Lm, Lm]
    return np.unravel_index(int(state), dimsizes)

# Auxiliary function to print a sequence of states
def printTraj(seq):
    ss = ""
    for st in seq:
        ss += printState(st) + "\n"

    return ss

# Auxiliary function to print a state
def printState(state):
    if type(state) in [list,tuple]:
        l = state[0]
        s = state[1]
    else:
        l,s = state2observ(state)

    return "%d (%d)" % (l, s)

print(10, state2observ('10'))
print(22, state2observ('22'))


def prob_trajectory(chain, sequence):
	state, prob = chain

	probAcc= 1

	partida = int(sequence[0])
    
	for element in sequence[1:]:
		dest = int(element)
		probAcc *= prob[partida][dest]
		partida = dest

	return probAcc

def stationary_dist(chain):
	states, prob = chain
   
	eigvals, eigvecs = np.linalg.eig(prob.T)
	
	index = np.argmin(np.abs(eigvals - 1))

	stationary = np.real(eigvecs[:, index])

	stationary = stationary / np.sum(stationary) #Normalized dist

	return stationary.reshape(1,-1)

def compute_dist(chain, dist, n):
    

	steps, prob = chain

	probAfterSteps = np.linalg.matrix_power(prob,n)

	dist = np.dot(dist,probAfterSteps)

	return dist

def simulate(chain, initDist, N):
	states, prob = chain
	

	traj = []
    
	current = str(np.random.choice(states, p=initDist.flatten()))
	traj.append(current)

	for _ in range(1, int(N)):

		current_idx = int(current)

		current = str(np.random.choice(states, p=prob[current_idx]))

		traj.append(current)


	return tuple(traj)


import numpy.random as rnd

def simulate_game(chain, initDist):
    
	wins = 0    
	
	traj = simulate(chain, initDist , 10000)  
	for element in traj:
		if int(element)>89:
			wins +=1
		
	return wins/10000

def sumArray(array):
	return np.sum(array[0,90:])

# Sampling Approach

rnd.seed(42)

chainGo_02 = load_chain('StopITSpider02.npy', 1)

chainGo_02Ns = len(chainGo_02[0])
initRandDist = rnd.random((1, chainGo_02Ns))
initRandDist = initRandDist / np.sum(initRandDist)

print(f'Mean for chain with rain 20, Always go: {simulate_game(chainGo_02, initRandDist)*100}%')

chain75_02 = load_chain('StopITSpider02.npy',.75)
print(f'Mean for chain with rain 20, 75 go: {simulate_game(chain75_02, initRandDist)*100}%')


chainGo_04 = load_chain('StopITSpider04.npy', 1)
chainGo_04Ns = len(chainGo_04[0])
initRandDist = rnd.random((1, chainGo_04Ns))
initRandDist = initRandDist / np.sum(initRandDist)

print(f'Mean for chain with rain 40, Always GO: {simulate_game(chainGo_04, initRandDist)*100}%')


chain75_04 = load_chain('StopITSpider04.npy', .75)
print(f'Mean for chain with rain 40, 75 Go: {simulate_game(chain75_04, initRandDist)*100}%')


#Analytical Approach
stat_chainGo_02 = stationary_dist(chainGo_02)
stat_chain75_02 = stationary_dist(chain75_02)
stat_chainGo_04 = stationary_dist(chainGo_04)
stat_chain75_04 = stationary_dist(chain75_04)


print(f'Stationary of rain 20, always go: {sumArray(stat_chainGo_02)*100}%')
print(f'Stationary of rain 20, 75 go: {sumArray(stat_chain75_02)*100}%')
print(f'Stationary of rain 40, always go: {sumArray(stat_chainGo_04)*100}%')
print(f'Stationary of rain 40, 75 go: {sumArray(stat_chain75_04)*100}%')

#In the sampling approach, the metric we used to evaluate which was the faster chain was % of steps, in a trjectory of size 10000, that the state was equal to winning sate (90:99)
#In the analytical approach, we summed the probabilites of the stationary distribution related to the winning states.

# In the sampling approach, with 20% rain, the best chain was the one with Gamma = 1.0 and with rain 40%, the best was Gamma = .75
# In the analytical approach, we verified the same results!


