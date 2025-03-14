
s = "Hello, world!"
print(s)





import math

a = 0


while a < 2 * math.pi: 
    print(a, "radians correspond to", a * 180 / math.pi, "degrees.")
    a = a + 0.5

a = input("Please insert a number:\n>> ")

for i in range(5):
    a = math.sqrt(float(a))
    print("Next square root:", a)

if a > 1:
    print(a, "is larger than 1.") 
else: 
    print(a, "is smaller than or equal to 1.")





def square(x):
    return x * x

print(square(2))





print("The variable s is accessible here:", s)


import numpy as np

A1 = np.array([[1, 2, 3], [4, 5, 6]])
print("2 x 3 array of numbers:")
print(A1)
print("This array is of dimension", A1.shape)


A2 = np.eye(3)
print("3 x 3 identity:")
print(A2)
print("This array is of dimension", A2.shape)


A3 = np.zeros((2, 3))
print("2 x 3 array of zeros:")
print(A3)
print("This array is of dimension", A3.shape)


A4 = np.ones(4);
print("4 x 0 array of ones (note how there is no second dimension):")
print(A4)
print("This array is of dimension", A4.shape)







A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("3 x 3 matrix:")
print(A)


print(A.dtype)

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float64')
print(A)
print(A.dtype)


B = np.arange(1,4)
print("Vector with all numbers between 1 and 3:")
print(B)


C = np.diag(B)
print("Diagonal matrix built from the vector B:")
print(C)





D = A + np.eye(3)
print("A + I:")
print(D)



E = np.dot(A, B.T)
print("A * B':")
print(E)



F = np.linalg.inv(D)
print("inv(D):")
print(F)




G = np.append([1, 2, 3], A)
print("Append matrix A to vector [1, 2, 3]:")
print(G)





H1 = np.append(A, [[10, 11, 12]], axis = 0)
H2 = np.append(A, [[4], [7], [10]], axis = 1)
print("Append [10, 11, 12] to A:")
print(H1)

print("Append [[4], [7], [10]] to A:")
print(H2)





print("A[0]:", A[0])
print("A[1]:", A[1])
print("A[1, 2]:", A[1, 2])  # More efficient
print("A[0][2]:", A[0][2])  # Less efficient






print("A[1:2,0:1]:", A[1:2,0:1])



print("A[:-2,::2]:", A[:-2,::2]) 

I = np.arange(10, 1, -1)
print("Vector I, with numbers between 10 and 1:")
print(I)





print("I[[3, 3, 1, 8]]:", I[[3, 3, 1, 8]])


print("I[np.array([3, 3, -3, 8])]:", I[np.array([3, 3, -3, 8])])


print("I[np.array([[1, 1], [2, 3]])]:", I[np.array([[1, 1], [2, 3]])])





































































import numpy.random as rnd
import time

A = rnd.rand(1000,1000)
B = rnd.rand(1000,1000);
C = np.zeros((1000,1000));

t = time.time()

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        C[i, j] = A[i, j] + B[i, j]
    
t1 = time.time() - t

t = time.time()
C = A + B;
t2 = time.time() - t

print("Computation time with cycle (in seconds):", t1)
print("Computation time with numpy operation (in seconds):", t2)
























import time
import numpy as np;
array1 = np.random.rand(100000,1)
array2 = np.random.rand(100000,1)
resArray = np.zeros((100000,))

t1 = time.time()
for i in range(100000):
    resArray[i] = array1[i] + array2[i]
t2 = time.time()

t3= time.time()
resArray = array1 + array2
t4 = time.time()


print(f'TIme between start and finish loop: {t2-t1}')
print(f'TIme between start and finish vectorized: {t4-t3}')




























import numpy as np
import math as m

pArray = np.empty((1001,))
vArray = np.empty((1001,))

pArray[0] = -m.pi/3 + 0.6
vArray[0] = 0

for i in range (1000):
    v_new = vArray[i] - (1/400) * m.cos(3 * pArray[i]) + (1/1000)

    v_new = max(-0.07, min(0.07, v_new))

    p_new = pArray[i] + v_new
    
    p_new = max(-1.2, min(0.6, p_new))
    
    vArray[i+1] = v_new
    pArray[i+1] = p_new


print(pArray, vArray)

















import numpy as np
import math as m

pArrayRandom = np.empty((1001,))
vArrayRandom = np.empty((1001,))
values = [1,0,-1]
uArray = np.random.choice(values, (10001,), p=[0.7,0.2,0.1])

pArrayRandom[0] = -m.pi/3 + 0.6
vArrayRandom[0] = 0

for i in range (1000):
    v_new = vArrayRandom[i] - (1/400) * m.cos(3 * pArrayRandom[i]) + (uArray[i]/1000)

    v_new = max(-0.07, min(0.07, v_new))

    p_new = pArrayRandom[i] + v_new
    
    p_new = max(-1.2, min(0.6, p_new))
    
    vArrayRandom[i+1] = v_new
    pArrayRandom[i+1] = p_new

print(pArrayRandom, vArrayRandom)












%matplotlib inline
import matplotlib.pyplot as plt


x = 100 * rnd.rand(100, 1)
y = 2 * x + 10 * rnd.randn(100, 1)



X = np.append(x, np.ones((100,1)), axis = 1)

f_est = np.dot(np.linalg.pinv(X), y)
y_est = np.dot(X, f_est)



plt.figure()
plt.plot(x, y_est)
plt.plot(x, y, 'x')

plt.xlabel('Input X');
plt.ylabel('Output Y');

plt.title('Linear regression');





























%matplotlib inline
import matplotlib.pyplot as plt

x = np.arange(1001)
y = pArray

plt.figure()
plt.plot(x,y)
plt.xlabel('Input X');
plt.ylabel('Output Y');
plt.title('u=1')

x = np.arange(1001)
y_rand = pArrayRandom

plt.figure()
plt.plot(x,y_rand)
plt.xlabel('Input X')
plt.ylabel('Output y_rand')
plt.title('u=rand')
















import numpy as np
import math as m

pArray = np.empty((1001,))
vArray = np.empty((1001,))

pArray[0] = -m.pi/3 + 0.6
vArray[0] = 0

for i in range (1000):
    
    u_t = -1 if pArray[i] > -0.85 else 1

    v_new = vArray[i] - (1/400) * m.cos(3 * pArray[i]) + (u_t/1000)

    v_new = max(-0.07, min(0.07, v_new))

    p_new = pArray[i] + v_new
    
    p_new = max(-1.2, min(0.6, p_new))
    
    vArray[i+1] = v_new
    pArray[i+1] = p_new



x = np.arange(1001)
y = pArray

plt.figure()
plt.plot(x,y)
plt.xlabel('Input X');
plt.ylabel('Output Y');
plt.title('u=1 until p=-0.85')


































