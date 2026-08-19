
A courier company operates a fleet of delivery trucks that make deliveries to different parts of the city. The trucks are equipped with GPS tracking devices that record the location of each truck at regular intervals. The locations are divided into three regions: downtown, the suburbs, and the countryside. The following table shows the probabilities of a truck transitioning between these regions at each time step:

| Current region | Probability of transitioning to downtown | Probability of transitioning to the suburbs | Probability of transitioning to the countryside |
|----------------|--------------------------------------------|-----------------------------------------------|------------------------------------------------|
| Downtown       | 0.3                                      | 0.4                                           | 0.3                                            |
| Suburbs        | 0.2                                      | 0.5                                           | 0.3                                            |
| Countryside    | 0.4                                      | 0.3                                           | 0.3                                            |

1. If a truck is currently in the suburbs, what is the probability that it will be in the downtown region after two time steps? [1.5p]
2. If a truck is currently in the suburbs, what is the probability that it will be in the downtown region **the first time** after two time steps? [1.5p]
3. Is this Markov chain irreducible? [1.5p]
4. What is the stationary distribution? [1.5p]
5. Advanced question: What is the expected number of steps until the first time one enters the downtown region having started in the suburbs region. Hint: to get within 1 decimal point, it is enough to compute the probabilities for hitting times below 30. [2p]

 ```{python}
# Part 1
import numpy as np

# Transition matrix
P = np.array([
    [0.3, 0.4, 0.3],  # Downtown
    [0.2, 0.5, 0.3],  # Suburbs
    [0.4, 0.3, 0.3]   # Countryside
])

# Compute two-step transition matrix
P2 = np.linalg.matrix_power(P, 2)

print("Two-step transition matrix P^2:")
print(P2)

# Probability of going from Suburbs (row 1) to Downtown (col 0) in 2 steps
prob_suburbs_to_downtown = P2[1, 0]

print("\nProbability (Suburbs → Downtown in 2 steps):", prob_suburbs_to_downtown)

# Fill in the answer to part 1 below as a decimal number
problem1_p1 = 0.28


# Part 2
#Suburbs → Downtown the first time in 2 steps
#Region indeces
downtown = 0
suburbs = 1
countryside = 2
#path1: Suburbs -> Countryside -> Downtown
path1 = P[suburbs, countryside] * P[countryside, downtown]
#path2: Suburbs -> Suburbs -> Downtown
path2 = P[suburbs, suburbs] * P[suburbs, downtown]
#path3: Suburbs -> Downtown -> Downtown INVALID PATH
#Total probability
total_probability = path1 + path2
print("\nTotal Probability (Suburbs → Downtown the first time in 2 steps):", total_probability)
# Fill in the answer to part 2 below as a decimal number
problem1_p2 = 0.22

# Part 4
"""
#SOLUTION2
# To find the stationary distribution π, we solve πP = π
# This can be rewritten as (P^T - I)π = 0 with the
# constraint that the sum of the entries in π is 1.
A = np.transpose(P) - np.eye(3)
# Append the constraint that the sum of the entries in π is 1
A = np.vstack([A, np.ones(3)])
b = np.array([0, 0, 0, 1])  # Right
# Solve the linear system
problem1_stationary = np.linalg.lstsq(A, b, rcond=None)[0]
print("\nStationary distribution π:")
print(problem1_stationary)

# Fill in the answer to part 4 below
# the answer should be a numpy array of length 3
# make sure that the entries sums to 1!
#problem1_stationary = XXX
"""


#SOLUTION2

# Compute eigenvalues and eigenvectors of P^T
eigenvalues, eigenvectors = np.linalg.eig(P.T)

# Find index of eigenvalue 1
idx = np.argmin(np.abs(eigenvalues - 1))

# Corresponding eigenvector
stationary = eigenvectors[:, idx].real

# Normalize to make it a probability distribution
stationary = stationary / stationary.sum()

problem1_stationary =np.array([stationary[0], stationary[1], stationary[2]])
print("Stationary distribution:", problem1_stationary)
#print("Downtown     :", stationary[0])
#print("Suburbs      :", stationary[1])
#print("Countryside  :", stationary[2])


# Part 5
#SOLUTION1
# The expected number of steps to return to a state i is given by 1/π[i]
expected_steps_downtown = 1 / problem1_stationary[downtown]
print("\nExpected number of steps to return to Downtown:", expected_steps_downtown)


# Fill in the answer to part 5 below
# That is, the expected number of steps as a decimal number
#problem1_ET = expected_steps_downtown


#SOLUTION2
# Expected hitting time to state 0 (Downtown)
# h(0)=0, and for i in {1,2}: h(i) = 1 + sum_j P[i,j]*h(j)
# Rearrange for states {1,2}: (I - Q) h = 1
Q = P[1:, 1:]                 # transitions among non-downtown states {1,2}
I = np.eye(Q.shape[0])
ones = np.ones(Q.shape[0])

h = np.linalg.solve(I - Q, ones)   # h[0]=h(Suburbs), h[1]=h(Countryside)

print("E[T | start=Suburbs]     =", h[0])
print("E[T | start=Countryside] =", h[1])
problem1_ET = h[0]

#SOLUTION3
"""
# Make Downtown absorbing (once you hit it, you stay there)
P_abs = P.copy()
P_abs[0] = [1.0, 0.0, 0.0]

# Start in Suburbs: distribution over states at time 0
v = np.array([0.0, 1.0, 0.0])

# E[T] = sum_{t>=0} P(T>t). Truncate at 30 steps (t = 0..29).
E_approx = 0.0
for t in range(30):
    prob_hit_by_t = v[0]           # since Downtown is absorbing, this is P(T <= t)
    prob_survive = 1.0 - prob_hit_by_t  # P(T > t)
    E_approx += prob_survive
    v = v @ P_abs                  # advance one step

print("Approx E[T] using t<30 =", E_approx)
"""



