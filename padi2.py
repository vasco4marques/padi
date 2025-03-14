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

p_policePlay=p_play
c_policePlay = c[:,0]

J_play = np.dot(np.linalg.inv(np.identity(12)- .9 * p_policePlay),c_policePlay)

print(f'JPlay AT 2 - {J_play[2]}')

p_policeStop = p_play
p_policeStop[2] = p_stop[2]

c_policeStop = c[:,0]
c_policeStop[2] = c[2][1]

J_stop = np.dot(np.linalg.inv(np.identity(12)- .9*p_policeStop),c_policeStop)

print(f'Jstop AT 2 - {J_stop[2]}')




