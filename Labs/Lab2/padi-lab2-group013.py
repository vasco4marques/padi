import numpy as np
import numpy.random as rand
import time

def load_mdp(fname, gamma):
	"""
	Builds an MDP model from the provided file.	
    
	:param fname: Name of the file containing the MDP information
	:type: str
	:param gamma: Discount
	:type: float
	:returns: tuple (tuple, tuple, tuple, nd.array, float)
	"""
	
	data = np.load(fname)

	P_stop = data[0]
	P_go   = data[1]  
    
	num_levels = len(data[0])
      
	X = tuple(str(level) for level in range(num_levels))
    
	A = ("St", "Go")
    
	P = (np.array(P_stop), np.array(P_go))
    
	num_states = len(X)
	num_actions = len(A)
    
	c = np.ones((num_states, num_actions))
    
	top_states = [ i for i in range(90,100) ]
	for i in top_states:
		c[i, :] = 0.0
    
	return (X, A, P, c, gamma)
    
def noisy_policy(mdp, a, eps):
	"""
	Builds a noisy policy around action a for a given MDP.

	:param mdp: MDP description
	:type: tuple
    :param a: main action for the policy
    :type: integer
    :param eps: noise level
    :type: float
    :return: nd.array
    """
    
	(X, A, P, c, gamma) = mdp
	
	num_states = len(X)
	num_actions = len(A)

	pol = np.full((num_states, num_actions), eps/(num_actions-1))

	pol[:,a] = 1. - eps

	return pol

def evaluate_pol(mdp, pol):
	"""
	Computes the cost-to-go function for a given policy in a given MDP.
	:param mdp: MDP description
	:type: tuple
	:param pol: Policy to be evaluated
	:type: nd.array
	:returns: nd.array
	"""
    
	X, A, P, c, gamma = mdp
	num_states = len(X)
	num_actions = len(A)
    
	c_pol = np.sum(pol * c, axis=1).reshape(num_states, 1)
    
	P_pol = np.zeros((num_states, num_states))
	for a in range(num_actions):
		P_pol += np.multiply(pol[:, a].reshape(num_states, 1), P[a])
    
	I = np.eye(num_states)
	v = np.dot(np.linalg.inv(I - gamma * P_pol), c_pol)

	return v

def value_iteration(M):
    
	X,A,P,C, gamma = M
	
	J = np.zeros((len(X), 1))
	err = 1.0
	eps = 1e-8
	k = 0

	now = time.time()

	while err > eps:
		Q = np.zeros((len(X), len(A)))

		for a in range(len(A)):
			Q[:, a, None] = C[:, a, None] + gamma * P[a].dot(J)

		Jnew = np.min(Q, axis=1, keepdims=True)

		err = np.linalg.norm(J - Jnew)
        
		J = Jnew
		k += 1

	done = time.time()
	print(f'Execution time: {np.round(done - now,3)} seconds.')
	print(f'Done after {k} iterations.')
    
	return J

def policy_iteration(mdp):
	"""
	Computes the optimal policy for a given MDP.

	:param mdp: MDP description
	:type: tuple
	:returns: nd.array
	"""
	X,A,P,C,gamma = mdp

	pol = np.ones((len(X), len(A))) / len(A)
	quit = False
	niter = 0

	now = time.time()

	while not quit:
		Q = np.zeros((len(X), len(A)))

		cpi = np.sum(C * pol, axis=1, keepdims=True)
		Ppi = pol[:, 0, None] * P[0]

		for a in range(1, len(A)):
			Ppi += pol[:, a, None] * P[a]

		J = np.linalg.inv(np.eye(len(X)) - gamma * Ppi).dot(cpi)

		for a in range(len(A)):
			Q[:, a, None] = C[:, a, None] + gamma * P[a].dot(J)

		Qmin = np.min(Q, axis=1, keepdims=True)
		pnew = np.isclose(Q, Qmin, atol=1e-8, rtol=1e-8).astype(int)
		pnew = pnew / pnew.sum(axis=1, keepdims=True)

		quit = (pol == pnew).all()

		pol = pnew
		niter += 1

	done = time.time()
	print(f'Execution time: {np.round(done - now,3)} seconds.')
	print(f'Done after {niter} iterations.')
	return np.round(pol,3)

NRUNS = 100 

def simulate(mdp, pol, x0, length=10000):
	"""
	Estimates the cost-to-go for a given MDP, policy and state.

	:param mdp: MDP description
	:type: tuple
	:param pol: policy to be simulated
	:type: nd.array
	:param x0: initial state
	:type: int
	:returns: float
	"""
	X, A, P, C, gamma = mdp
	num_states = len(X)
	num_actions = len(A)
    
	total_cost = 0.0
    
	for run in range(NRUNS):
		current_state = x0
		trajectory_cost = 0.0

		for t in range(length):
			action = np.random.choice(num_actions, p=pol[current_state])
            
			immediate_cost = C[current_state, action]
			trajectory_cost += (gamma**t) * immediate_cost
            
			next_state = np.random.choice(num_states, p=P[action][current_state])
			current_state = next_state
            
		total_cost += trajectory_cost

	estimated_cost = total_cost / NRUNS
	return estimated_cost 