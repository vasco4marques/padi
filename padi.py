import numpy as np

# Given matrices
p_play = np.array([
    [0.2, 0.4, 0.4, 0,   0,   0,   0,   0,   0,   0,   0,   0],
    [0.2, 0,   0.4, 0.4, 0,   0,   0,   0,   0,   0,   0,   0],
    [0.2, 0,   0,   0.4, 0.4, 0,   0,   0,   0,   0,   0,   0],
    [0.2, 0,   0,   0,   0.4, 0.4, 0,   0,   0,   0,   0,   0],
    [0.2, 0,   0,   0,   0,   0.8, 0,   0,   0,   0,   0,   0],
    [0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0],
    [0,   0,   0,   0,   0,   0,   0.2, 0.4, 0.4, 0,   0,   0],
    [0,   0,   0,   0,   0,   0,   0.2, 0,   0.4, 0.4, 0,   0],
    [0,   0,   0,   0,   0,   0,   0,   0,   0.2, 0.4, 0.4, 0],
    [0,   0,   0,   0,   0,   0,   0,   0,   0.2, 0,   0.4, 0.4],
    [0,   0,   0,   0,   0,   0,   0,   0,   0.2, 0,   0, 0.8],
    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1]
])

p_stop = np.array([
    [1,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0],
    [1,0,0,0,0,0,1,0,0,0,0,0],
    [1,0,0,0,0,0,1,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,1,0,0,0],
    [1,0,0,0,0,0,0,0,1,0,0,0],
    [1,0,0,0,0,0,0,0,1,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,1]
])
    
c = np.array([
    [1, 1],  # State 0
    [1, 0],  # State 1
    [1, 0],  # State 2
    [1, 0],  # State 3
    [1, 0],  # State 4
    [0, 0],  # State 5 (Terminal)
    [1, 1],  # State 6
    [1, 0],  # State 7
    [1, 1],  # State 8
    [1, 0],  # State 9
    [1, 0],  # State 10
    [0, 0]   # State 11 (Terminal)
])

gamma = 0.9  # Discount factor


def compute_optimal_cost():
    """Computes the optimal     cost-to-go J* for all states."""
    num_states = c.shape[0]
    J = np.zeros(num_states)  # Initialize J with zeros
    threshold = 1e-6  # Convergence threshold

    while True:
        J_new = np.zeros(num_states)

        for s in range(num_states):
            if s == 5 or s == 11:  # Terminal states
                J_new[s] = 0
                continue
            
            # Compute the cost for each action
            J_play = c[s, 0] + gamma * np.dot(p_play[s], J)
            J_stop = c[s, 1] + gamma * np.dot(p_stop[s], J)

            # Take the minimum cost action
            J_new[s] = min(J_play, J_stop)

        # Check for convergence
        if np.max(np.abs(J_new - J)) < threshold:
            break
        
        J = J_new  # Update J for next iteration

    return J


# Precompute J* for all states
J_opt = compute_optimal_cost()


def calcOptJ(x):
    """Returns the optimal cost-to-go J*(x) for a given state x."""
    return J_opt[x]

def calcJ(state):
    """Calculates J(s, play) and J(s, stop) for a given state."""
    if state == 5 or state == 11:  # Terminal states
        return (0, 0)

    # Compute cost-to-go for each action
    J_play = c[state, 0] + gamma * np.dot(p_play[state], J_opt)
    J_stop = c[state, 1] + gamma * np.dot(p_stop[state], J_opt)

    return (J_play, J_stop)


# Example usage:
print(calcOptJ(11))
print(calcOptJ(10))
print(calcOptJ(9))
print(calcOptJ(8))
print(calcOptJ(7))
print(calcOptJ(6))
print("----------")
print(calcOptJ(5))  # Get the optimal cost-to-go from state 0
print(calcOptJ(4))  # Get the optimal cost-to-go from state 1
print(calcOptJ(3))
print(calcOptJ(2))
print(calcOptJ(1))
print(calcOptJ(0))
print(calcJ(2))