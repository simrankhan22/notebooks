
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


---
```

<a id="examjanuary-2022-problem5-markovian-travel"></a>
## ExamJanuary_2022.PROBLEM5 – Markovian travel
## Markovian travel

The dataset `Travel Dataset - Datathon 2019` is a simulated dataset designed to mimic real corporate travel systems -- focusing on flights and hotels. The file is at `data/flights.csv` in the same folder as `Exam.ipynb`, i.e. you can use the path `data/flights.csv` from the notebook to access the file.

1. [2p] In the first code-box
    1. Load the csv from file `data/flights.csv`
    2. Fill in the value of the variables as specified by their names.
2. [2p] In the second code-box your goal is to estimate a Markov chain transition matrix for the travels of these users. For example, if we enumerate the cities according to alphabetical order, the first city `'Aracaju (SE)'` would correspond to $0$. Each row of the file corresponds to one flight, i.e. it has a starting city and an ending city. We model this as a stationary Markov chain, i.e. each user's travel trajectory is a realization of the Markov chain, $X_t$. Here, $X_t$ is the current city the user is at, at step $t$, and $X_{t+1}$ is the city the user travels to at the next time step. This means that to each row in the file there is a corresponding pair $(X_{t},X_{t+1})$. The stationarity assumption gives that for all $t$ there is a transition density $p$ such that $P(X_{t+1} = y | X_t = x) = p(x,y)$ (for all $x,y$). The transition matrix should be `n_cities` x `n_citites` in size.
3. [2p] Use the transition matrix to compute out the stationary distribution.
4. [2p] Given that we start in 'Aracaju (SE)' what is the probability that after 3 steps we will be back in 'Aracaju (SE)'?


print("Approx E[T] using t<30 =", E_approx)
"""

```
import pandas as pd
#read the fole
df = pd.read_csv("data/flights.csv")
#fill value of variables
number_of_observations = df.shape[0]
number_of_userCodes = df["userCode"].nunique()
number_of_cities = len(pd.unique(pd.concat([df["from"], df["to"]])))
#display variables 
number_of_cities, number_of_userCodes, number_of_observations



# This is a very useful function that you can use for part 2. You have seen this before when parsing the
# pride and prejudice book.

def makeFreqDict(myDataList):
    '''Make a frequency mapping out of a list of data.

    Param myDataList, a list of data.
    Return a dictionary mapping each unique data value to its frequency count.'''

    freqDict = {} # start with an empty dictionary

    for res in myDataList:
        if res in freqDict: # the data value already exists as a key
                freqDict[res] = freqDict[res] + 1 # add 1 to the count using sage integers
        else: # the data value does not exist as a key value
            freqDict[res] = 1 # add a new key-value pair for this new data value, frequency 1

    return freqDict # return the dictionary created


import numpy as np
import pandas as pd

# Build cities
cities = pd.concat([df["from"], df["to"]])
unique_cities = sorted(set(cities))
n_cities = len(unique_cities)


# Count the different transitions
#transitions = XXX # A list containing tuples ex: ('Aracaju (SE)','Rio de Janeiro (RJ)') of all transitions in the text
#transition_counts = XXX # A dictionary that counts the number of each transition
transitions = list(zip(df["from"], df["to"]))

transition_counts = makeFreqDict(transitions)

# ex: ('Aracaju (SE)','Rio de Janeiro (RJ)'):4
#indexToCity = XXX # A dictionary that maps the n-1 number to the n:th unique_city,
indexToCity = {i: city for i, city in enumerate(unique_cities)}
# ex: 0:'Aracaju (SE)'
#cityToIndex = XXX # The inverse function of indexToWord,
cityToIndex = {city: i for i, city in enumerate(unique_cities)}
# ex: 'Aracaju (SE)':0

# Part 3, finding the maximum likelihood estimate of the transition matrix

#transition_matrix = XXX # a numpy array of size (n_cities,n_cities)
transition_matrix = np.zeros((n_cities, n_cities))
# The transition matrix should be ordered in such a way that
# p_{'Aracaju (SE)','Rio de Janeiro (RJ)'} = transition_matrix[cityToIndex['Aracaju (SE)'],cityToIndex['Rio de Janeiro (RJ)']]
# and represents the probability of travelling Aracaju (SE)->Rio de Janeiro (RJ)

# Make sure that the transition_matrix does not contain np.nan from division by zero for instance



for (city_from, city_to), count in transition_counts.items():
    i = cityToIndex[city_from]
    j = cityToIndex[city_to]
    transition_matrix[i, j] += count

# Normalize rows
row_sums = transition_matrix.sum(axis=1, keepdims=True)
transition_matrix = np.divide(
    transition_matrix,
    row_sums,
    where=row_sums != 0
)
print(transition_matrix)


# This should be a numpy array of length n_cities which sums to 1 and is all positive
# To find the stationary distribution π, we solve πP = π
# This can be rewritten as (P^T - I)π = 0 with the
# constraint that the sum of the entries in π is 1.
n = transition_matrix.shape[0]

A = transition_matrix.T - np.eye(n)   # (9, 9)
# Append the constraint that the sum of the entries in π is 1
A = np.vstack([A, np.ones(n)])        # (10, 9)

b = np.zeros(n + 1)                   # length 10
b[-1] = 1                             # sum(pi)=1 constraint
# Solve the linear system
stationary_distribution_problem5 = np.linalg.lstsq(A, b, rcond=None)[0]

print(stationary_distribution_problem5)
print("sum =", stationary_distribution_problem5.sum())



# Compute the return probability for part 3 of problem 5
# Compute tree-step transition matrix
i = cityToIndex['Aracaju (SE)']
transition_matrix_3 = np.linalg.matrix_power(transition_matrix, 3)
print(transition_matrix_3)
return_probability = transition_matrix_3[i, i]
print(return_probability_problem5)



# Once you have created all your functions, you can make a small test here to see
# what would be generated from your model.
import numpy as np

start = np.zeros(shape=(n_cities,1))
start[cityToIndex['Aracaju (SE)'],0] = 1

current_pos = start
for i in range(10):
    random_word_index = np.random.choice(range(n_cities),p=current_pos.reshape(-1))
    current_pos = np.zeros_like(start)
    current_pos[random_word_index] = 1
    print(indexToCity[random_word_index],end='->')
    current_pos = (current_pos.T@transition_matrix).T

```


