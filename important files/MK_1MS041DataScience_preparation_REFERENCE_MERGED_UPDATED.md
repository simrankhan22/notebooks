# 1MS041 Data Science – Exam2026 Preparation Notebook

https://github.com/BlueJ1/IntroDSExamMaterial?tab=readme-ov-file

https://bluej1.github.io/IntroDSExamMaterial/

https://github.com/datascience-intro/1MS041-2025/

<a id="index-problem-theme-oriented"></a>
## Index (problem/theme oriented)

### Assignments
**Assignment 1**
- [Assignment1.PROBLEM1.Conceptual questions: AI/ML basics & data science reading](#assignment-1-problem-1)
- [Assignment1.PROBLEM2.Shell/CSV basics: inspecting files & pandas read_csv/head](#assignment-1-problem-2)
- [Assignment1.PROBLEM3.CSV parsing with Python csv module (reader, headers, rows)](#assignment-1-problem-3)
- [Assignment1.PROBLEM4.Discrete probability: Binomial model & conditional probabilities](#assignment-1-problem-4)
- [Assignment1.PROBLEM5.Concentration inequalities: exponential tails & sample size](#assignment-1-problem-5)

**Assignment 2**
- [Assignment2.PROBLEM1.Markov chains: transition matrices, stationary distribution, hitting/return times](#assignment-2-problem-1)
- [Assignment2.PROBLEM2.MLE via optimization: likelihood, gradients, constraints](#assignment-2-problem-2)
- [Assignment2.PROBLEM3.Hypothesis testing & confidence intervals](#assignment-2-problem-3)
- [Assignment2.PROBLEM4.Random variable generation and transformation](#assignment-2-problem-4)

**Assignment 3**
- [Assignment3.PROBLEM1.Linear regression: design matrix, fit, metrics](#assignment-3-problem-1)
- [Assignment3.PROBLEM2.Classification with SVM: margins, hinge loss, SVC](#assignment-3-problem-2)
- [Assignment3.PROBLEM3.Regression models: trees, evaluation, model comparison](#assignment-3-problem-3)

**Assignment 4**
- [Assignment4.PROBLEM1.Pipelines: preprocessing + model, train/validation workflow](#assignment-4-problem-1)

### Exams
- [ExamJanuary_2022.PROBLEM1.Probability warmup](#examjanuary-2022-problem1-probability-warmup)
- [ExamJanuary_2022.PROBLEM2.Random variable generation and transformation](#examjanuary-2022-problem2-random-variable-generation-and-transformation)
- [ExamJanuary_2022.PROBLEM3.Concentration of measure](#examjanuary-2022-problem3-concentration-of-measure)
- [ExamJanuary_2022.PROBLEM4.SMS spam filtering](#examjanuary-2022-problem4-sms-spam-filtering)
- [ExamJanuary_2022.PROBLEM5.Markovian travel](#examjanuary-2022-problem5-markovian-travel)
- [ExamJanuary_2022.PROBLEM6.Black box testing](#examjanuary-2022-problem6-black-box-testing)
- [ExamJanuary_2023.PROBLEM1.Markov chains](#examjanuary-2023-problem1-markov-chains)
- [ExamJanuary_2023.PROBLEM2.Linear regression](#examjanuary-2023-problem2-linear-regression)
- [ExamJanuary_2023.PROBLEM3.Count regression (visits)](#examjanuary-2023-problem3-count-regression-visits)
- [Exam2024.PROBLEM1.Random variable generation: rejection & inversion sampling](#exam2024-problem1-random-variable-generation-rejection-inversion-sampling)
- [Exam2024.PROBLEM2.Logistic regression: calibration & predictive intervals (spam example)](#exam2024-problem2-logistic-regression-calibration-predictive-intervals-spam-example)
- [Exam2024.PROBLEM3.Markov chains: classification of properties & stationary behavior](#exam2024-problem3-markov-chains-classification-of-properties-stationary-behavior)

### Python Reference
- [Python Reference section](#python-reference)
- [`print(*objects, sep=' ', end='\n')`](#pyref-print)
- [`len(obj)`](#pyref-len)
- [`range(start, stop, step)`](#pyref-range)
- [`enumerate(iterable, start=0)`](#pyref-enumerate)
- [`isinstance(obj, cls)`](#pyref-isinstance)
- [`open(path, mode='r', encoding=None)`](#pyref-open)
- [`sum(iterable)`](#pyref-sum)
- [`math.log(x)`](#pyref-math-log)
- [`math.comb(n, k)`](#pyref-math-comb)
- [`pd.read_csv(path)`](#pyref-pd-read-csv)
- [`df.head(n=5)`](#pyref-df-head)
- [`csv.reader(file_obj)`](#pyref-csv-reader)
- [`optimize.minimize(fun, x0, method=...)`](#pyref-optimize-minimize)
- [`SVC(kernel='linear')`](#pyref-svc)
- [`Pipeline(steps=[...])`](#pyref-pipeline)
- [`DecisionTreeRegressor(max_depth=..., random_state=...)`](#pyref-decisiontreeregressor)
- [`np.random.default_rng(seed=None)`](#pyref-np-random-default-rng-seed-none)
- [`np.mean(a)`](#pyref-np-mean-a)
- [`np.sqrt(x)`](#pyref-np-sqrt-x)
- [`np.log(x)`](#pyref-np-log-x)
- [`np.array(obj)`](#pyref-np-array-obj)
- [`np.zeros(shape)`](#pyref-np-zeros-shape)
- [`np.ones(shape)`](#pyref-np-ones-shape)
- [`np.eye(n)`](#pyref-np-eye-n)
- [`np.vstack(tup)`](#pyref-np-vstack-tup)
- [`np.divide(a, b, where=...)`](#pyref-np-divide-a-b-where)
- [`np.linalg.lstsq(A, b, rcond=None)`](#pyref-np-linalg-lstsq-a-b-rcond-none)
- [`np.linalg.matrix_power(A, k)`](#pyref-np-linalg-matrix-power-a-k)
- [`pd.concat(objs, axis=0)`](#pyref-pd-concat-objs-axis-0)
- [`pd.unique(values)`](#pyref-pd-unique-values)
- [`text.lower()`](#pyref-text-lower)
- [`text.count(sub)`](#pyref-text-count-sub)
- [`zip(*iterables)`](#pyref-zip-iterables)


# Assignment 1 for Course 1MS041
Make sure you pass the `# ... Test` cells and
 submit your solution notebook in the corresponding assignment on the course website. You can submit multiple times before the deadline and your highest score will be used.

---

<a id="assignment-1-problem-1"></a>
## Assignment 1, PROBLEM 1
Maximum Points = 3



Given that you are being introduced to data science it is important to bear in mind the true costs of AI, a highly predictive family of algorithms used in data engineering sciences:

Read the 16 pages of [ai-anatomy-publication.pdf](http://www.anatomyof.ai/img/ai-anatomy-publication.pdf) with the highly detailed [ai-anatomy-map.pdf](https://anatomyof.ai/img/ai-anatomy-map.pdf) of [https://anatomyof.ai/](https://anatomyof.ai/), "Anatomy of an AI System" By Kate Crawford and Vladan Joler (2018).  The first problem in ASSIGNMENT 1 is a trivial test of your reading comprehension.


Answer whether each of the following statements is `True` or `False` *according to the authors* by appropriately replacing `Xxxxx` coresponding to `TruthValueOfStatement0a`, `TruthValueOfStatement0b` and `TruthValueOfStatement0c`, respectively, in the next cell to demonstrate your reading comprehension.

1. `Statement0a =` *Each small moment of convenience (provided by Amazon's Echo) – be it answering a question, turning on a light, or playing a song – requires a vast planetary network, fueled by the extraction of non-renewable materials, labor, and data.*
2. `Statement0b =` *The Echo user is simultaneously a consumer, a resource, a worker, and a product* 
3. `Statement0c =` *Many of the assumptions about human life made by machine learning systems are narrow, normative and laden with error. Yet they are inscribing and building those assumptions into a new world, and will increasingly play a role in how opportunities, wealth, and knowledge are distributed.*

```python

# Replace Xxxxx with True or False; Don't modify anything else in this cell!

TruthValueOfStatement0a = True

TruthValueOfStatement0b = True

TruthValueOfStatement0c = True
```

---
#### Local Test for Assignment 1, PROBLEM 1
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.

```python
# Test locally to ensure an acceptable answer, True or False
try:
    assert(isinstance(TruthValueOfStatement0a, bool)) 
    assert(isinstance(TruthValueOfStatement0b, bool)) 
    assert(isinstance(TruthValueOfStatement0c, bool))
except:
    print("Try again. You are not writing True or False for your answers.")
else:
    print("Good, you have answered either True or False. Hopefully they are the correct answers!")
```

---

<a id="assignment-1-problem-2"></a>
## Assignment 1, PROBLEM 2
Maximum Points = 2


Evaluate the following cells by replacing `X` with the right command-line option to `head` in order to find the first four lines of the csv file `data/final.csv`

```
%%sh
man head

HEAD(1)                   BSD General Commands Manual                  HEAD(1)

NAME
     head -- display first lines of a file

SYNOPSIS
     head [-n count | -c bytes] [file ...]

DESCRIPTION
     This filter displays the first count lines or bytes of each of the speci-
     fied files, or of the standard input if no files are specified.  If count
     is omitted it defaults to 10.

     If more than a single file is specified, each file is preceded by a
     header consisting of the string ``==> XXX <=='' where ``XXX'' is the name
     of the file.

EXIT STATUS
     The head utility exits 0 on success, and >0 if an error occurs.

SEE ALSO
     tail(1)

HISTORY
     The head command appeared in PWB UNIX.

BSD                              June 6, 1993                              BSD
```

```python
#%%sh
#head -X data/final.csv
import pandas as pd
fl = pd.read_csv('data/final.csv')
fl.head(4)
```

```python
#with open("data/final.csv", "r", encoding="utf-8") as f:
#    lines = f.readlines()

#line_1_final = lines[0].strip().split(",")   # line=1
#line_2_final = lines[1].strip().split(",")   # line=2

line_1_final= "region,municipality,district,party,votes"
line_2_final= "Blekinge län,Karlshamn,0 - Centrala Asarum,S,519"
#print("line_1_final:", line_1_final)
#print("line_2_final:", line_2_final)
```

---
#### Local Test for Assignment 1, PROBLEM 2
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.

```python
# Evaluate this cell locally to make sure you have the answer as a string
try:
    assert(type(line_1_final) == str)
    print("Good! You have answered as a string for line 1. Hopefully it is the correct!")
except AssertionError:
    print("Try Again. You should answer with a string.")
try:
    assert(type(line_2_final) == str)
    print("Good! You have answered as a string for line 2. Hopefully it is the correct!")
except AssertionError:
    print("Try Again. You should answer with a string.")
```

---

<a id="assignment-1-problem-3"></a>
## Assignment 1, PROBLEM 3
Maximum Points = 3


In this assignment the goal is to parse the `final.csv` file from the previous problem.

1. Read the file `data/final.csv` and parse it using the `csv` package and store the result as follows

the `header` variable contains a list of names all as strings

the `data` variable should be a list of lists containing all the rows of the csv file

```python

import csv

data = []
with open('data/final.csv', newline='', encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)  # Read the header row
    #print(f'Header: {header}')
    #row_count = sum(1 for row in reader)
    #print(f'The number of rows in the file is: {row_count}')
    for row in reader:
       data.append(row)  # Append each row to the list
#print(data[:100])  # Print the first 100 rows to verify


#header = XXX
#data = XXX
```

---
#### Local Test for Assignment 1, PROBLEM 3
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.

```python

# Evaluate this cell locally to make sure you have the answer in the right format
try:
    assert(type(header) == list)
    print("Good! You have the header as a list. Hopefully it is the correct!")
except AssertionError:
    print("Try Again. You should answer with a list.")
try:
    types = set([type(a) for a in header])
    assert((len(types) == 1) and (list(types)[0] == str))
    print("Good! You have the header as a list of strings. Hopefully it is the correct!")
except AssertionError:
    print("Try Again. You should answer with a list of strings.")
try:
    assert(type(data) == list)
    print("Good! You have the data as a list. Hopefully it is the correct!")
except AssertionError:
    print("Try Again. You should answer with a list.")
try:
    types = set([type(a) for a in data])
    assert((len(types) == 1) and (list(types)[0] == list))
    print("Good! You have the data as a list of lists. Hopefully it is the correct!")
except AssertionError:
    print("Try Again. You should answer with a list of lists.")
try:
    types = set(sum([[type(d) for d in t] for t in data[:1]],[]))
    assert((len(types) == 1) and (list(types)[0] == str))
    print("Good! You have the data as a list of lists of strings. Hopefully it is the correct!")
except AssertionError:
    print("Try Again. You should answer with a list of lists of strings.")
```

---

<a id="assignment-1-problem-4"></a>
## Assignment 1, PROBLEM 4
Maximum Points = 8

<a id="students-passing-exam-sample-exam-problem"></a>

## Students passing exam (Sample exam problem)
Let's say we have an exam question which consists of $10$ yes/no questions. 
From past performance of similar students, a randomly chosen student will know the correct answer to $N \sim \text{binom}(10,6/10)$ questions. Furthermore, we assume that the student will guess the answer with equal probability to each question they don't know the answer to, i.e. given $N$ we define $Z \sim \text{binom}(10-N,1/2)$ as the number of correctly guessed answers. Define $Y = N + Z$, i.e., $Y$ represents the number of total correct answers.

We are interested in setting a deterministic threshold $T$, i.e., we would pass a student at threshold $T$ if $Y \geq T$. Here $T \in \{0,1,2,\ldots,10\}$.

1. [5p] For each threshold $T$, compute the probability that the student *knows* less than $5$ correct answers given that the student passed, i.e., $N < 5$. Put the answer in `problem11_probabilities` as a list.
2. [3p] What is the smallest value of $T$ such that if $Y \geq T$ then we are 90\% certain that $N \geq 5$?

```python
# Hint the PMF of N is p_N(k) where p_N is
#import scipy  
from math import comb
#from scipy.special import binom as binomial

n = 10                  # number of questions
p = 6/10                # probability student knows a question
p_guess = 1/2          # probability of guessing correctly
p_N = lambda k: comb(10,k)*((1-p)**(10-k))*((p)**k)
problem11_probabilities1 = []

for T in range(n+1):
    numerator = 0.0   # P(N<5 and Y>=T)
    denominator = 0.0 # P(Y>=T)
    
    for N in range(n+1):
        # Distribution of Z given N
        for z in range(n-N+1):
            prob = p_N(N) * comb(n-N, z) * (p_guess**z) * ((1-p_guess)**((n-N)-z))
            Y = N + z
            if Y >= T:
                denominator += prob
                if N < 5:
                    numerator += prob
    
    cond_prob = numerator / denominator if denominator > 0 else 0
    problem11_probabilities1.append(cond_prob)

for T, prob in enumerate(problem11_probabilities1):
    print(f"T={T}: P(N<5 | Y>=T) = {prob:.6f}")
```

```python
# Part 1: 
# replace XXX to represent P(N < 5) for T = [0,1,2,...,10], i.e. your answer should be a list
# of length 11.
problem11_probabilities = [0.166239,0.166239,0.166235,0.166174,0.165517,0.160894,0.144453,0.112230,0.073212,0.040585,0.019728]
#problem11_probabilities = "0.166239,0.166239,0.166235,0.166174,0.165517,0.160894,0.144453,0.112230,0.073212,0.040585,0.019728"
#print(problem11_probabilities)
```

```python

# Part 2: Give an integer between 0 and 10 which is the answer to 2.
problem12_T = 7
```

---

<a id="assignment-1-problem-5"></a>
## Assignment 1, PROBLEM 5
Maximum Points = 8

<a id="concentration-of-measure-sample-exam-problem"></a>

## Concentration of measure (Sample exam problem)

As you recall, we said that concentration of measure was simply the phenomenon where we expect that the probability of a large deviation of some quantity becoming smaller as we observe more samples: [0.4 points per correct answer]

1. Which of the following will exponentially concentrate, i.e. for some $C_1,C_2,C_3,C_4 $ 
$$
    P(Z - \mathbb{E}[Z] \geq \epsilon) \leq C_1 e^{-C_2 n \epsilon^2} \vee C_3 e^{-C_4 n (\epsilon+1)} \enspace .
$$

    1. The empirical variance of i.i.d. random variables with finite mean?
    2. The empirical variance of i.i.d. sub-Gaussian random variables?
    3. The empirical variance of i.i.d. sub-Exponential random variables?
    4. The empirical mean of i.i.d. sub-Gaussian random variables?
    5. The empirical mean of i.i.d. sub-Exponential random variables?
    6. The empirical mean of i.i.d. random variables with finite variance?
    7. The empirical third moment of i.i.d. random variables with finite sixth moment?
    8. The empirical fourth moment of i.i.d. sub-Gaussian random variables?
    9. The empirical mean of i.i.d. deterministic random variables?
    10. The empirical tenth moment of i.i.d. Bernoulli random variables?

2. Which of the above will concentrate in the weaker sense, that for some $C_1$
$$
    P(Z - \mathbb{E}[Z] \geq \epsilon) \leq \frac{C_1}{n \epsilon^2}?
$$

```python

# Answers to part 1, which of the alternatives exponentially concentrate, answer as a list
# i.e. [1,4,5] that is example 1, 4, and 5 concentrate
problem3_answer_1 = [2,4,5,8,9,10]
```

```python

# Answers to part 2, which of the alternatives concentrate in the weaker sense, answer as a list
# i.e. [1,4,5] that is example 1, 4, and 5 concentrate
problem3_answer_2 = [2,3,4,5,6,7,8,9,10]
```


---


<a id="mk-assignment-2-ver3"></a>
## MK_Assignment_2_ver3


# Assignment 2 for Course 1MS041
Make sure you pass the `# ... Test` cells and
 submit your solution notebook in the corresponding assignment on the course website. You can submit multiple times before the deadline and your highest score will be used.

---

<a id="assignment-2-problem-1"></a>
## Assignment 2, PROBLEM 1
Maximum Points = 8


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

```python
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
```

```python
# import matplotlib.pyplot as plt
```

```python
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
```

```python
# Part 3
# Check if the Markov chain is irreducible
#Every entry in the transition matrix P is positive (> 0)
#This means any state can reach any other state in just 1 step
#Therefore, all states communicate with each other
#By definition, the chain is irreducible
# Fill in the answer to part 3 below as a boolean
problem1_irreducible = True
```

```python
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
```

```python
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
```

---

<a id="assignment-2-problem-2"></a>
## Assignment 2, PROBLEM 2
Maximum Points = 4


Use the **Multi-dimensional Constrained Optimisation** example (in `07-Optimization.ipynb`) to numerically find the MLe for the mean and variance parameter based on `normallySimulatedDataSamples`, an array obtained by a specific simulation of $30$ IID samples from the $Normal(10,2)$ random variable.

Recall that $Normal(\mu, \sigma^2)$ RV has the probability density function given by:

$$
f(x ;\mu, \sigma) = \displaystyle\frac{1}{\sigma\sqrt{2\pi}}\exp\left(\frac{-1}{2\sigma^2}(x-\mu)^2\right)
$$

The two parameters, $\mu \in \mathbb{R} := (-\infty,\infty)$ and $\sigma \in (0,\infty)$, are sometimes referred to as the location and scale parameters.

You know that the log likelihood function for $n$ IID samples from a Normal RV with parameters $\mu$ and $\sigma$ simply follows from $\sum_{i=1}^n \log(f(x_i; \mu,\sigma))$, based on the IID assumption. 

NOTE: When setting bounding boxes for $\mu$ and $\sigma$ try to start with some guesses like $[-20,20]$ and $[0.1,5.0]$ and make it larger if the solution is at the boundary. Making the left bounding-point for $\sigma$ too close to $0.0$ will cause division by zero Warnings. Other numerical instabilities can happen in such iterative numerical solutions to the MLe. You need to be patient and learn by trial-and-error. You will see the mathematical theory in more details in a future course in scientific computing/optimisation. So don't worry too much now except learning to use it for our problems.

```python

import numpy as np
from scipy import optimize
# do NOT change the next three lines
np.random.seed(123456) # set seed
# simulate 30 IID samples drawn from Normal(10,2)RV
normallySimulatedDataSamples = np.random.normal(10,2,30) 

# define the negative log likelihoo function you want to minimise by editing XXX
def negLogLklOfIIDNormalSamples(parameters):
    '''return the -log(likelihood) of normallySimulatedDataSamples with mean and var parameters'''
    mu_param=parameters[0]
    sigma_param=parameters[1]
    
    x = normallySimulatedDataSamples
    n = x.size

    rss = np.sum((x - mu_param)**2)      # residual sum of squares
    nll = n*np.log(sigma_param) + 0.5*n*np.log(2*np.pi) + 0.5*rss/(sigma_param**2)
    return nll
    #XXX
    #XXX # add more or less lines as you need
    #return XXX 

# you should only change XXX below and not anything else
parameter_bounding_box=((-20.0, 20.0), (0.1, 5.0)) # specify the constraints for each parameter - some guess work...
initial_arguments = np.array([5.0, 1.0]) # point in 2D to initialise the minimize algorithm
result_problem2_opt = optimize.minimize(negLogLklOfIIDNormalSamples, initial_arguments, bounds=parameter_bounding_box, method="L-BFGS-B") 
# call the minimize method above finally! you need to play a bit to get initial conditions and bounding box ok
result_problem2_opt
```

---

<a id="assignment-2-problem-3"></a>
## Assignment 2, PROBLEM 3
Maximum Points = 4



Derive the maximum likelihood estimate for $n$ IID samples from a random variable with the following probability density function:
$$
f(x; \lambda) = \frac{1}{24} \lambda^5 x^4 \exp(-\lambda x), \qquad \text{ where, } \lambda>0, x > 0
$$

You can solve the MLe by hand (using pencil paper or using key-strokes). Present your solution as the return value of a function called `def MLeForAssignment2Problem3(x)`, where `x` is a list of $n$ input data points.

```python

# do not change the name of the function, just replace XXX with the appropriate expressions for the MLe
#def MLeForAssignment2Problem3(x):
#    '''write comment of what this function does'''
#    XXX 
#    XXX
#    return XXX
def MLeForAssignment2Problem3(x):
    """
    Returns the MLE of lambda for the pdf
    f(x; lambda) = (1/24) * lambda^5 * x^4 * exp(-lambda*x), x>0, lambda>0.
    x: list/iterable of n positive samples.
    """
    if not x:
        raise ValueError("x must contain at least one observation.")
    s = sum(x)
    if s <= 0:
        raise ValueError("All observations must be positive.")
    n = len(x)
    return 5.0 * n / s   # equivalently: 5.0 / (s / n)
print(MLeForAssignment2Problem3([1,2,3,4,5]))  # example usage
```

---

<a id="assignment-2-problem-4"></a>
## Assignment 2, PROBLEM 4
Maximum Points = 8

<a id="random-variable-generation-and-transformation"></a>

## Random variable generation and transformation

The purpose of this problem is to show that you can implement your own sampler, this will be built in the following three steps:

1. [2p] Implement a Linear Congruential Generator where you tested out a good combination (a large $M$ with $a,b$ satisfying the Hull-Dobell (Thm 6.8)) of parameters. Follow the instructions in the code block.
2. [2p] Using a generator construct random numbers from the uniform $[0,1]$ distribution.
3. [4p] Using a uniform $[0,1]$ random generator, generate samples from 

$$p_0(x) = \frac{\pi}{2}|\sin(2\pi x)|, \quad x \in [0,1] \enspace .$$

Using the **Accept-Reject** sampler (**Algorithm 1** in TFDS notes) with sampling density given by the uniform $[0,1]$ distribution.

```python
import numpy as np

def problem4_LCG(size=None, seed=0):
    """
    A linear congruential generator that generates pseudo random numbers according to size.
    
    Parameters
    -------------
    size : an integer denoting how many samples should be produced
    seed : the starting point of the LCG, i.e. u0 in the notes.
    
    Returns
    -------------
    out : a list of the pseudo random numbers
    """

    # Parameters satisfying Hull-Dobell theorem
    m = 2**48
    a = 25214903917
    c = 11

    # Initialize
    x = seed
    out = []

    # Generate pseudo-random numbers
    for _ in range(size):
        x = (a * x + c) % m
        out.append(x / m)  # Normalize to [0,1)

    return out
print(problem4_LCG(size=10, seed=0))  # example usage
```

```python

def problem4_uniform(generator=None, period = 1, size=None, seed=0):
    """
    Takes a generator and produces samples from the uniform [0,1] distribution according
    to size.
    
    Parameters
    -------------
    generator : a function of type generator(size,seed) and produces the same result as problem1_LCG, i.e. pseudo random numbers in the range {0,1,...,period-1}
    period : the period of the generator
    seed : the seed to be used in the generator provided
    size : an integer denoting how many samples should be produced
    
    Returns
    --------------
    out : a list of the uniform pseudo random numbers
    """
    
    # XXX
    
    # return XXX
    if size is None:
        raise ValueError("size must be provided")
    if generator is None:
        raise ValueError("You must provide a generator function (e.g. problem1_LCG)")

    # Step 1: Generate pseudo-random integers using the provided generator
    random_integers = generator(size, seed)

    # Step 2: Convert to uniform [0,1) values
    uniform_numbers = [x / float(period) for x in random_integers]

    return uniform_numbers
#print(problem4_uniform(generator=problem4_LCG, period=2**48, size=10, seed=0))  # example usage
```

```python

def problem4_accept_reject(uniformGenerator=None, n_iterations=None, seed=0):
    """
    Takes a generator that produces uniform pseudo random [0,1] numbers 
    and produces samples from (pi/2)*abs(sin(x*2*pi)) using an Accept-Reject
    sampler with the uniform distribution as the proposal distribution.
    Runs n_iterations
    
    Parameters
    -------------
    generator : a function of the type generator(size,seed) that produces uniform pseudo random
    numbers from [0,1]
    seed : the seed to be used in the generator provided
    n_iterations : an integer denoting how many attempts should be made in the accept-reject sampler
    
    Returns
    --------------
    out : a list of the pseudo random numbers with the specified distribution
    """
    
    #XXX
    
    #return XXX

    if n_iterations is None:
        raise ValueError("n_iterations must be provided")

    # Default Uniform(0,1) generator if none was passed
    if uniformGenerator is None:
        def uniformGenerator(size, seed):
            rng = np.random.default_rng(seed)
            return rng.random(size).tolist()

    # We need two uniforms per attempt: one for x, one for the acceptance test
    u = uniformGenerator(2 * n_iterations, seed)
    out = []
    idx = 0

    # M = sup_x p0(x)/q(x) with q(x)=1 on [0,1]. Since max |sin(2πx)| = 1, M = π/2.
    # Accept if U <= p0(x) / (M*q(x)) = |sin(2πx)|.
    for _ in range(n_iterations):
        x = u[idx]; idx += 1           # proposal from Uniform(0,1)
        u_acc = u[idx]; idx += 1       # uniform for acceptance test
        if u_acc <= abs(np.sin(2 * np.pi * x)):
            out.append(x)

    return out
```

---
#### Local Test for Assignment 2, PROBLEM 4
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.

```python

# If you managed to solve all three parts you can test the following code to see if it runs
# you have to change the period to match your LCG though, this is marked as XXX.
# It is a very good idea to check these things using the histogram function in sagemath
# try with a larger number of samples, up to 10000 should run

print("LCG output: %s" % problem4_LCG(size=10, seed = 1))

period = 2**48

print("Uniform sampler %s" % problem4_uniform(generator=problem4_LCG, period = period, size=10, seed=1))

uniform_sampler = lambda size,seed: problem4_uniform(generator=problem4_LCG, period = period, size=size, seed=seed)

print("Accept-Reject sampler %s" % problem4_accept_reject(uniformGenerator = uniform_sampler,n_iterations=20,seed=1))
```

```python

# If however you did not manage to implement either part 1 or part 2 but still want to check part 3, you can run the code below

def testUniformGenerator(size,seed):
    import random
    random.seed(seed)
    
    return [random.uniform(0,1) for s in range(size)]

print("Accept-Reject sampler %s" % problem4_accept_reject(uniformGenerator=testUniformGenerator, n_iterations=20, seed=1))
```


---


<a id="mk-assignment-3-ver2"></a>
## MK_Assignment_3_ver2


# Assignment 3 for Course 1MS041
Make sure you pass the `# ... Test` cells and
 submit your solution notebook in the corresponding assignment on the course website. You can submit multiple times before the deadline and your highest score will be used.

---

<a id="assignment-3-problem-1"></a>
## Assignment 3, PROBLEM 1
Maximum Points = 8


Download the updated data folder from the course github website or just download directly the file [https://github.com/datascience-intro/1MS041-2025/blob/main/notebooks/data/smhi.csv](https://github.com/datascience-intro/1MS041-2025/blob/main/notebooks/data/smhi.csv) from the github website and put it inside your data folder, i.e. you want the path `data/smhi.csv`. The data was aquired from SMHI (Swedish Meteorological and Hydrological Institute) and constitutes per hour measurements of wind in the Uppsala Aut station. The data consists of windspeed and direction. Your goal is to load the data and work with it a bit. The code you produce should load the file as it is, please do not alter the file as the autograder will only have access to the original file.

The file information is in Swedish so you need to use some translation service, for instance `Google translate` or ChatGPT.

1. [2p] Load the file, for instance using the `csv` package. Put the wind-direction as a numpy array and the wind-speed as another numpy array.
2. [2p] Use the wind-direction (see [Wikipedia](https://en.wikipedia.org/wiki/Wind_direction)) which is an angle in degrees and convert it into a point on the unit circle **which is the direction the wind is blowing to** (compare to definition of radians [Wikipedia](https://en.wikipedia.org/wiki/Radian)). Store the `x_coordinate` as one array and the `y_coordinate` as another. From these coordinates, construct the wind-velocity vector.
3. [2p] Calculate the average wind velocity and convert it back to direction and compare it to just taking average of the wind direction as given in the data-file.
4. [2p] The wind velocity is a $2$-dimensional random variable, calculate the empirical covariance matrix which should be a numpy array of shape (2,2).

For you to wonder about, is it more likely for you to have headwind or not when going to the university in the morning.

```python
import csv
import numpy as np

csv_path = "data/smhi.csv"  

wind_dir_list = []   # to store wind direction values
wind_speed_list = []



with open(csv_path, "r", encoding="utf-8") as f:
    # Step 1: skip metadata until real header is found
    for line in f:
        if line.startswith("Datum;"):
            header = line.strip().split(";")
            break

    # Step 2: create DictReader using the detected header
    reader = csv.DictReader(f, delimiter=";", fieldnames=header)

    # Step 3: read all data rows
    for row in reader:
        # ignore empty rows or metadata rows after header
        if not row["Datum"] or row["Datum"].startswith("2024") is False:
            continue

        try:
            # Step 4: extract and convert values
            wind_dir = int(row["Vindriktning"])
            wind_speed = float(row["Vindhastighet"])

            wind_dir_list.append(wind_dir)
            wind_speed_list.append(wind_speed)

        except (ValueError, KeyError):
            # skip rows that contain text instead of numbers
            continue

# Step 5: convert to numpy arrays
problem1_wind_direction = np.array(wind_dir_list, dtype=int)
problem1_wind_speed = np.array(wind_speed_list, dtype=float)
print("Wind direction array:", problem1_wind_direction)
print("Wind speed array:", problem1_wind_speed)
```

```python
# The wind direction is given as a compass direction in degrees (0-360)
# convert it to x and y coordinates using the standard mathematical convention


# Convert meteorological wind direction ("from") to mathematical ("to")
theta = np.radians(270-problem1_wind_direction)
#ver1 theta = np.radians(problem1_wind_direction + 180)
# Unit circle coordinates
problem1_wind_direction_x_coordinate = np.cos(theta)
problem1_wind_direction_y_coordinate = np.sin(theta)

# Wind velocity vector components
problem1_wind_velocity_x_coordinate = problem1_wind_speed* problem1_wind_direction_x_coordinate
problem1_wind_velocity_y_coordinate = problem1_wind_speed * problem1_wind_direction_y_coordinate

print("Wind velocity X components:", problem1_wind_velocity_x_coordinate)
print("Wind velocity Y components:", problem1_wind_velocity_y_coordinate)
```

```python

# Average velocity components
problem1_average_wind_velocity_x_coordinate = np.mean(
    problem1_wind_velocity_x_coordinate
)
problem1_average_wind_velocity_y_coordinate = np.mean(
    problem1_wind_velocity_y_coordinate
)

# Angle of the average velocity vector, in degrees (0–360)
problem1_average_wind_velocity_angle_degrees = np.degrees(
    np.arctan2(
        problem1_average_wind_velocity_y_coordinate,
        problem1_average_wind_velocity_x_coordinate,
    )
) % 360

print("Average wind direction (to):", problem1_average_wind_velocity_angle_degrees)

# Now: "just taking average of the wind direction as given in the data-file"
# (simple arithmetic mean of the direction values in degrees)
problem1_average_wind_direction_angle_degrees = (270-
    problem1_wind_direction.astype(float) % 360
).mean()


print(
    "Average wind direction from data (to):",
    problem1_average_wind_direction_angle_degrees,
)

# Finally: are they the same?
problem1_same_angle = False
```

```python
import numpy as np

vel_matrix = np.vstack((
    problem1_wind_velocity_x_coordinate,
    problem1_wind_velocity_y_coordinate
))

problem1_wind_velocity_covariance_matrix = np.cov(vel_matrix, bias=False)

print("Covariance matrix:", problem1_wind_velocity_covariance_matrix)
```

---

<a id="assignment-3-problem-2"></a>
## Assignment 3, PROBLEM 2
Maximum Points = 8


For this problem you will need the [pandas](https://pandas.pydata.org/) package and the [sklearn](https://scikit-learn.org/stable/) package. Inside the `data` folder from the course website you will find a file called `indoor_train.csv`, this file includes a bunch of positions in (X,Y,Z) and also a location number. The idea is to assign a room number (Location) to the coordinates (X,Y,Z).

1. [2p] Take the data in the file `indoor_train.csv` and load it using pandas into a dataframe `df_train`
2. [3p] From this dataframe `df_train`, create two numpy arrays, one `Xtrain` and `Ytrain`, they should have sizes `(1154,3)` and `(1154,)` respectively. Their `dtype` should be `float64` and `int64` respectively.
3. [3p] Train a Support Vector Classifier, `sklearn.svc.SVC`, on `Xtrain, Ytrain` with `kernel='linear'` and name the trained model `svc_train`.

To mimic how [kaggle](https://www.kaggle.com/) works, the Autograder has access to a hidden test-set and will test your fitted model.

```python
#import sys
#print(sys.executable)
#!{sys.executable} -m pip install scikit-learn


import pandas as pd
import sklearn as skl
#Assign path
train_csv_path = "data/indoor_train.csv"
#Read the CSV file into a DataFrame
df_train = pd.read_csv(train_csv_path)
df_train.head()
#print("Columns:", df_train.columns.tolist())
```

```python
# clean column names if they have BOM or extra whitespace
df_train.columns = df_train.columns.str.replace('\ufeff', '', regex=False).str.strip()

# Create Xtrain as (n_samples, 3) float64 and Ytrain as (n_samples,) int64
Xtrain = df_train[['Position X', 'Position Y', 'Position Z']].to_numpy(dtype='float64')  
Ytrain = df_train['Location'].to_numpy(dtype='int64')
print("Columns:", df_train.columns.tolist())
print("Xtrain:", Xtrain)
print("Ytrain:", Ytrain)
```

```python
from sklearn.svm import SVC

svc_train = SVC(kernel="linear")  # linear kernel
svc_train.fit(Xtrain, Ytrain)
print("SVC model trained.")


#y_pred = svc_train.predict(Xtest)

#from sklearn.metrics import accuracy_score
#problem2_test_accuracy = accuracy_score(Xtest, y_pred)
#print("Test accuracy:", problem2_test_accuracy)
```

---

<a id="assignment-3-problem-3"></a>
## Assignment 3, PROBLEM 3
Maximum Points = 8


Let us build a proportional model ($\mathbb{P}(Y=1 \mid X) = G(\beta_0+\beta \cdot X)$ where $G$ is the logistic function) for the spam vs not spam data. Here we assume that the features are presence vs not presence of a word, let $X_1,X_2,X_3$ denote the presence (1) or absence (0) of the words $("free", "prize", "win")$.

1. [2p] Load the file `data/spam.csv` and create two numpy arrays, `problem3_X` which has shape **(n_texts,3)** where each feature in `problem3_X` corresponds to $X_1,X_2,X_3$ from above, `problem3_Y` which has shape **(n_texts,)** and consists of a $1$ if the email is spam and $0$ if it is not. Split this data into a train-calibration-test sets where we have the split $40\%$, $20\%$, $40\%$, put this data in the designated variables in the code cell.

2. [2p] Follow the calculation from the lecture notes where we derive the logistic regression and implement the final loss function inside the class `ProportionalSpam`. You can use the `Test` cell to check that it gives the correct value for a test-point.

3. [2p] Train the model `problem3_ps` on the training data. The goal is to calibrate the probabilities output from the model. Start by creating a new variable `problem3_X_pred` (shape `(n_samples,1)`) which consists of the predictions of `problem3_ps` on the calibration dataset. Then train a calibration model using `sklearn.tree.DecisionTreeRegressor`, store this trained model in `problem3_calibrator`. Recall that calibration error is the following for a fixed function $f$
$$
    \sqrt{\mathbb{E}[|\mathbb{E}[Y \mid f(X)] - f(X)|^2]}.
$$

4. [2p] Use the trained model `problem3_ps` and the calibrator `problem3_calibrator` to make final predictions on the testing data, store the prediction in `problem3_final_predictions`.

```python
import numpy as np
import pandas as pd

#Assign path
spam_csv_path = "data/spam.csv"
#Read the CSV file into a DataFrame
#ver1 df_spam = pd.read_csv(spam_csv_path, header=None, encoding='utf-8', encoding_errors='ignore')
df_spam = pd.read_csv(spam_csv_path, encoding='latin-1')
df_spam.head()

#Create numpy arrays problem3_X and problem3_Y
# column names not known in advance
label_col = df_spam.columns[0]   # "v1"  -> ham/spam
text_col  = df_spam.columns[1]   # "v2"  -> message text

# Build problem3_X with features for X1="free", X2="prize", X3="win"
words = ["free", "prize", "win"]

# Number of texts in the dataset and number of words to search for
n_texts = len(df_spam) # number of rows in the DataFrame
problem3_X = np.zeros((n_texts, len(words)), dtype=int) # initialize feature matrix

# Fill the feature matrix
for j, w in enumerate(words):
    problem3_X[:, j] = df_spam[text_col].str.contains(    # search for whole word w
        fr"\b{w}\b", case=False, na=False                   # regex=True
    ).astype(int)     # convert boolean to int (0/1)

# Build problem3_Y as 1 for "spam" and 0 for "ham"
problem3_Y = (df_spam[label_col].str.lower() == "spam").astype(int).to_numpy()

# Split the data into training (40%), calibration (20%), and test (40%) sets
n_train = int(0.4 * n_texts)
n_calib = int(0.2 * n_texts)
n_test = n_texts - n_train - n_calib

problem3_X_train = problem3_X[:n_train]
problem3_X_calib = problem3_X[n_train:n_train + n_calib]
problem3_X_test = problem3_X[n_train + n_calib:]
problem3_Y_train = problem3_Y[:n_train]
problem3_Y_calib = problem3_Y[n_train:n_train + n_calib]
problem3_Y_test = problem3_Y[n_train + n_calib:]
print(problem3_X_train.shape,problem3_X_calib.shape,problem3_X_test.shape,problem3_Y_train.shape,problem3_Y_calib.shape,problem3_Y_test.shape)
```

```python


class ProportionalSpam(object):
    def __init__(self):
        self.coeffs = None
        self.result = None
    
    # define the objective/cost/loss function we want to minimise
    def loss(self,X,Y,coeffs):
            import numpy as np

            # linear part  z_i = w0 + x_i·w
            z = np.dot(X, coeffs[1:]) + coeffs[0]

            # logistic function σ(z) = 1 / (1 + e^(−z))
            p = 1.0 / (1.0 + np.exp(-z))   # predicted probability P(y=1 | x)

            # negative log-likelihood / cross-entropy loss
            eps = 1e-15                    # to avoid log(0)
            p = np.clip(p, eps, 1 - eps)   # keep p in (0,1)
            loss = -np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p))

            return loss

    def fit(self,X,Y):
        import numpy as np
        from scipy import optimize

        #Use the f above together with an optimization method from scipy
        #to find the coefficients of the model
        opt_loss = lambda coeffs: self.loss(X,Y,coeffs)
        initial_arguments = np.zeros(shape=X.shape[1]+1)
        self.result = optimize.minimize(opt_loss, initial_arguments,method='cg')
        self.coeffs = self.result.x
    
    def predict(self,X):
        #Use the trained model to predict Y
        if (self.coeffs is not None):
            G = lambda x: np.exp(x)/(1+np.exp(x))
            return np.round(10*G(np.dot(X,self.coeffs[1:])+self.coeffs[0]))/10 # This rounding is to help you with the calibration
```

```python
# 1. Train ProportionalSpam on the training data
problem3_ps = ProportionalSpam()
problem3_ps.fit(problem3_X_train, problem3_Y_train)

# 2. Get predicted probabilities on the *calibration* set
#    problem3_X_pred must have shape (n_samples, 1)
raw_calib_pred = problem3_ps.predict(problem3_X_calib)   # shape (n_samples,)
problem3_X_pred = raw_calib_pred.reshape(-1, 1)        # shape (n_samples, 1)

# 3. Train a calibration model (DecisionTreeRegressor)
from sklearn.tree import DecisionTreeRegressor

problem3_calibrator = DecisionTreeRegressor(
    max_depth=3,        
    random_state=0
)
problem3_calibrator.fit(problem3_X_pred, problem3_Y_calib)


print("problem3_X_pred", problem3_X_pred.shape)
print("problem3_calibrator", problem3_calibrator)
#problem3_ps = XXX
#problem3_X_pred = XXX
#problem3_calibrator = XXX
```

```python
# 1. Raw probabilities from ProportionalSpam on the test set
test_raw_pred = problem3_ps.predict(problem3_X_test)      # shape: (n_test,)

# 2. Reshape to (n_test, 1) because the calibrator expects a 2D array
test_raw_pred_2d = test_raw_pred.reshape(-1, 1)

# 3. Calibrated final predictions
problem3_final_predictions = problem3_calibrator.predict(test_raw_pred_2d)

print ("Final predictions on test set:", problem3_final_predictions)
#problem3_final_predictions = XXX
```

---
#### Local Test for Assignment 3, PROBLEM 3
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.

```python
try:
    import numpy as np
    test_instance = ProportionalSpam()
    test_loss = test_instance.loss(np.array([[1,0,1],[0,1,1]]),np.array([1,0]),np.array([1.2,0.4,0.3,0.9]))
    assert (np.abs(test_loss-1.2828629432232497) < 1e-6)
    print("Your loss was correct for a test point")
except:
    print("Your loss was not correct on a test point")
```


---


<a id="mk-assignment-4-ver1"></a>
## MK_Assignment_4_ver1


# Assignment 4 for Course 1MS041
Make sure you pass the `# ... Test` cells and
 submit your solution notebook in the corresponding assignment on the course website. You can submit multiple times before the deadline and your highest score will be used.

---

<a id="assignment-4-problem-1"></a>
## Assignment 4, PROBLEM 1
Maximum Points = 24


    This time the assignment only consists of one problem, but we will do a more comprehensive analysis instead.

Consider the dataset `Corona_NLP_train.csv` that you can get from the course website [git](https://github.com/datascience-intro/1MS041-2024/blob/main/notebooks/data/Corona_NLP_train.csv). The data is "Coronavirus tweets NLP - Text Classification" that can be found on [kaggle](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification). The data has several columns, but we will only be working with `OriginalTweet`and `Sentiment`.

1. [3p] Load the data and filter out those tweets that have `Sentiment`=`Neutral`. Let $X$ represent the `OriginalTweet` and let 
    $$
        Y = 
        \begin{cases}
        1 & \text{if sentiment is towards positive}
        \\
        0 & \text{if sentiment is towards negative}.
        \end{cases}
    $$
    Put the resulting arrays into the variables $X$ and $Y$. Split the data into three parts, train/test/validation where train is 60% of the data, test is 15% and validation is 25% of the data. Do not do this randomly, this is to make sure that we all did the same splits (we are in this case assuming the data is IID as presented in the dataset). That is [train,test,validation] is the splitting layout.

2. [4p] There are many ways to solve this classification problem. The first main issue to resolve is to convert the $X$ variable to something that you can feed into a machine learning model. For instance, you can first use [`CountVectorizer`](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) as the first step. The step that comes after should be a `LogisticRegression` model, but for this to work you need to put together the `CountVectorizer` and the `LogisticRegression` model into a [`Pipeline`](https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline). Fill in the variable `model` such that it accepts the raw text as input and outputs a number $0$ or $1$, make sure that `model.predict_proba` works for this. **Hint: You might need to play with the parameters of LogisticRegression to get convergence, make sure that it doesn't take too long or the autograder might kill your code**
3. [3p] Use your trained model and calculate the precision and recall on both classes. Fill in the corresponding variables with the answer.
4. [3p] Let us now define a cost function
    * A positive tweet that is classified as negative will have a cost of 1
    * A negative tweet that is classified as positive will have a cost of 5
    * Correct classifications cost 0
    
    complete filling the function `cost` to compute the cost of a prediction model under a certain prediction threshold (recall our precision recall lecture and the `predict_proba` function from trained models). 

5. [4p] Now, we wish to select the threshold of our classifier that minimizes the cost, fill in the selected threshold value in value `optimal_threshold`.
6. [4p] With your newly computed threshold value, compute the cost of putting this model in production by computing the cost using the validation data. Also provide a confidence interval of the cost using Hoeffdings inequality with a 99% confidence.
7. [3p] Let $t$ be the threshold you found and $f$ the model you fitted (one of the outputs of `predict_proba`), if we define the random variable
    $$
        C = (1-1_{f(X)\geq t})Y+5(1-Y)1_{f(X) \geq t}
    $$
    then $C$ denotes the cost of a randomly chosen tweet. In the previous step we estimated $\mathbb{E}[C]$ using the empirical mean. However, since the threshold is chosen to minimize cost it is likely that $C=0$ or $C=1$ than $C=5$ as such it will have a low variance. Compute the empirical variance of $C$ on the validation set. What would be the confidence interval if we used Bennett's inequality instead of Hoeffding in point 6 but with the computed empirical variance as our guess for the variance?

```python
# Part 1

# Load the data from the file specified in the problem definition and make sure that it is loaded using
# the search path `data/Corona_NLP_train.csv`. This is to make sure the autograder and your computer have the same
# file path and can load the data correctly.

# Contrary to how many other problems are structured, this problem actually requires you to
# have X on the shape (n_samples, ) that is a 1-dimensional array. Otherwise it will cause a bunch
# of errors in the autograder or also in for instance CountVectorizer.

# Make sure that all your data is numpy arrays and not pandas dataframes or series.
import csv
import numpy as np

csv_path = "data/Corona_NLP_train.csv"  
#print("Loading data from:", csv_path)

X = np.array([])
Y = np.array([])

with open(csv_path, 'r', encoding='latin1') as file:
    reader = csv.DictReader(file)
    texts = []
    labels = []
    for row in reader:
        texts.append(row['OriginalTweet'])
        labels.append(row['Sentiment'])
    X = np.array(texts)
    Y = np.array(labels)

# Filter out Neutral sentiments
mask = Y != 'Neutral'
X = X[mask]
Y = Y[mask]

# Map sentiments to 0/1: 1 for positive (towards positive), 0 for negative
Y = np.where((Y == 'Positive') | (Y == 'Extremely Positive'), 1, 0)

# Split the data sequentially: train 60%, test 15%, validation 25%
n = len(X)
train_end = int(0.6 * n)
test_end = train_end + int(0.15 * n)

X_train = X[:train_end]
Y_train = Y[:train_end]
X_test = X[train_end:test_end]
Y_test = Y[train_end:test_end]
X_valid = X[test_end:]
Y_valid = Y[test_end:]
```

```python

# Part 2

# Train a machine learning model or pipeline that can take the raw strings from X and predict Y=0,1 depending on the
# sentiment of the tweet. Store the trained model in the variable `model`.
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Define the model pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression())
])
# Train the model
model.fit(X_train, Y_train)
# Verify outputs
probs = model.predict_proba(X_test)
print(probs[:5]) # Print the first 5 predicted probabilities
# The output should be a 2D numpy array of shape (n_samples, 2)
print("Output shape:", probs.shape)
```

```python

# Part 3

# Evaluate the model on the test set and calculate precision, and recall on both classes. Store the results in the
# variables `precision_0`, `precision_1`, `recall_0`, `recall_1`.
from sklearn.metrics import precision_score, recall_score
y_pred = model.predict(X_test)

precision_0 = precision_score(Y_test, y_pred, pos_label=0)
precision_1 = precision_score(Y_test, y_pred, pos_label=1)
recall_0 = recall_score(Y_test, y_pred, pos_label=0)
recall_1 = recall_score(Y_test, y_pred, pos_label=1)
# Verify outputs
print(f"Precision class 0: {precision_0}")
print(f"Precision class 1: {precision_1}")
print(f"Recall class 0: {recall_0}")
print(f"Recall class 1: {recall_1}")
```

```python

# Part 4

def cost(model,threshold,X,Y):
    # Hint, make sure that the model has a predict_proba method
    # think about how the decision is made based on the probabilities
    # and how the threshold can be used to make the decision.
    # For reference take a look at the lecture notes "Bayes classifier"
    # which contains how the decision is made based on the probabilities when the threshold is 0.5.
    
    # Fill in what is missing to compute the cost and return it
    # Note that we are interested in average cost
    """
    Average cost under asymmetric misclassification costs:
      - True positive predicted negative (FN): cost 1
      - True negative predicted positive (FP): cost 5
      - Correct: cost 0
    Decision rule: predict 1 if P(Y=1) >= threshold else 0
    """

    # Probability of class 1 (positive)
    p_pos = model.predict_proba(X)[:, 1]

    # Threshold decision
    y_pred = (p_pos >= threshold).astype(int)

    # Compute costs vectorized
    fn = (Y == 1) & (y_pred == 0)   # positive classified as negative
    fp = (Y == 0) & (y_pred == 1)   # negative classified as positive

    total_cost = fn.sum() * 1 + fp.sum() * 5

    # Average cost (as requested)
    return total_cost / len(Y)
# Verify outputs
example_threshold = 0.7
example_cost = cost(model, example_threshold, X_test, Y_test)
print(f"Cost at threshold {example_threshold}: {example_cost}")
```

```python

# Part 5

# Find the optimal threshold for the model on the test set. Store the threshold in the variable `optimal_threshold`
# and the cost at the optimal threshold in the variable `cost_at_optimal_threshold` evaluated on the test set.

# Candidate thresholds
thresholds = np.linspace(0.0, 1.0, 101)

costs = []

for t in thresholds:
    c = cost(model, t, X_valid, Y_valid)
    costs.append(c)

# Convert to numpy array
costs = np.array(costs)

# Index of minimum cost
best_idx = np.argmin(costs)

# Optimal threshold
optimal_threshold = thresholds[best_idx]
cost_at_optimal_threshold = costs[best_idx]
# Verify outputs
print(f"Optimal threshold: {optimal_threshold}")
print(f"Cost at optimal threshold: {cost_at_optimal_threshold}")
```

```python

# Part 6
# 1) Compute average validation cost at the chosen threshold
val_cost = cost(model, optimal_threshold, X_valid, Y_valid)  # average cost

# 2) Hoeffding 99% CI for the true (production) expected cost
delta = 0.01         # 99% confidence
a, b = 0.0, 5.0      # per-sample cost range
n = len(Y_valid)

epsilon = (b - a) * np.sqrt(np.log(2.0 / delta) / (2.0 * n))

ci_low  = max(a, val_cost - epsilon)
ci_high = min(b, val_cost + epsilon)

print("Optimal threshold:", optimal_threshold)
print("Validation average cost:", val_cost)
print("99% Hoeffding CI: [{:.6f}, {:.6f}]".format(ci_low, ci_high))
print("Half-width (epsilon):", epsilon)

cost_at_optimal_threshold_valid = val_cost
cost_interval_valid = (ci_low, ci_high)

assert(type(cost_interval_valid) == tuple)
assert(len(cost_interval_valid) == 2)
```

```python
# Part 7
import numpy as np
import math

def compute_C(model, t, X, Y):
    p_pos = model.predict_proba(X)[:, 1]
    I = (p_pos >= t).astype(int)
    C = (1 - I) * Y + 5 * (1 - Y) * I
    return C.astype(float)

C_val = compute_C(model, optimal_threshold, X_valid, Y_valid)

mean_C = float(np.mean(C_val))
var_C  = float(np.var(C_val, ddof=0))   # empirical variance (plug-in)

variance_of_C = var_C

def bennett_epsilon(n, v, c, delta=0.01):
    # Two-sided Bennett half-width for the mean
    if v <= 0 or n <= 0:
        return 0.0

    def h(u):
        return (1.0 + u) * math.log(1.0 + u) - u

    target = math.log(2.0 / delta)

    lo, hi = 0.0, c
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        u = (c * mid) / v
        lhs = (n * v / (c * c)) * h(u)
        if lhs >= target:
            hi = mid
        else:
            lo = mid
    return hi

delta = 0.01
a, b = 0.0, 5.0
c = b - a
n = len(C_val)

eps = bennett_epsilon(n=n, v=var_C, c=c, delta=delta)

ci_low  = max(a, mean_C - eps)
ci_high = min(b, mean_C + eps)

interval_of_C = (ci_low, ci_high)

assert isinstance(interval_of_C, tuple)
assert len(interval_of_C) == 2
```

# Exams (added)

This section contains full problem statements + solution code from the uploaded exam notebooks, reformatted for quick search.

<a id="examjanuary-2023-source-notebook"></a>
## ExamJanuary_2023 (source notebook)

```python
# Insert your anonymous exam ID as a string in the variable below
examID="test" \
""
```

---

<a id="examjanuary-2023-problem1-markov-chains"></a>
## ExamJanuary_2023.PROBLEM1 – Markov chains
Maximum Points = 14

A courier company operates a fleet of delivery trucks that make deliveries to different parts of the city. The trucks are equipped with GPS tracking devices that record the location of each truck at regular intervals. The locations are divided into three regions: downtown, the suburbs, and the countryside. The following table shows the probabilities of a truck transitioning between these regions at each time step:

| Current region | Probability of transitioning to downtown | Probability of transitioning to the suburbs | Probability of transitioning to the countryside |
|----------------|--------------------------------------------|-----------------------------------------------|------------------------------------------------|
| Downtown       | 0.3                                      | 0.4                                           | 0.3                                            |
| Suburbs        | 0.2                                      | 0.5                                           | 0.3                                            |
| Countryside    | 0.4                                      | 0.3                                           | 0.3                                            |

1. If a truck is currently in the suburbs, what is the probability that it will be in the downtown region after two time steps? [2p]
2. If a truck is currently in the suburbs, what is the probability that it will be in the downtown region **the first time** after two time steps? [2p]
3. Is this Markov chain irreducible? Explain your answer. [3p]
4. What is the stationary distribution? [3p]
5. Advanced question: What is the expected number of steps until the first time one enters the suburbs region having started in the downtown region. Hint: to get within 1 decimal point, it is enough to compute the probabilities for hitting times below 30. Motivate your answer in detail [4p]. You could also solve this question by simulation, but this gives you a maximum of [2p].

```python
# Part 1
import numpy as np
#define transition matrix
P=np.array([
[0.3, 0.4, 0.3],  #Downtown
[0.2, 0.5, 0.3],    #Suburbs
[0.4, 0.3, 0.3]   #Countryside 
])
#print the transition matrix
print ("The transition matrix:", P)
# Calculated trnasition matrix after 2 time steps
P2 = np.linalg.matrix_power(P, 2)
print("The transition matrix after 2 time steps:", P2)
# The probability of being in the downtown starting from suburbs after 2 time steps
prob_suburbs_to_downtown = P2[1, 0]
# Fill in the answer to part 1 below
problem1_p1 = prob_suburbs_to_downtown
print ("The probability of being in the downtown starting from suburbs after 2 time steps:", problem1_p1)
```

```python
# Part 2
# assign indices for transition matrix
downtown = 0
suburbs = 1
countryside = 2
# Sububs->Sububs->Downtown path
path1 = P[suburbs, suburbs] * P[suburbs, downtown]
#Suburbs->Downtown->Downtown INVALID PATH
#Suburbs->Countryside->Downtown
path2 =P[suburbs, countryside] * P[countryside, downtown]
#calculated probability that it will be in the downtown region **the first time** after two time steps
prob_to_downtown_1time_in_2_steps = path1+path2
# Fill in the answer to part 2 below
problem1_p2 = prob_to_downtown_1time_in_2_steps
print(prob_to_downtown_1time_in_2_steps)
```

```python
# Part 3

# Fill in the answer to part 3 below as a boolean
problem1_irreducible = True
```

<a id="part-3"></a>

## Part 3


Given Markov chain is irredusable, because all states communicate. In accordance to communicating states definition: Two states i and j communicate if each is reachable from the other

```python
# Part 4

# Part 4

#SOLUTION2
# To find the stationary distribution π, we solve πP = π
# This can be rewritten as (P^T - I)π = 0 with the
# constraint that the sum of the entries in π is 1.
#A = np.transpose(P) - np.eye(3)
# Append the constraint that the sum of the entries in π is 1
#A = np.vstack([A, np.ones(3)])
#b = np.array([0, 0, 0, 1])  # Right
# Solve the linear system
#problem1_stationary = np.linalg.lstsq(A, b, rcond=None)[0]
#print("\nStationary distribution π:")
#print(problem1_stationary)

# Fill in the answer to part 4 below
# the answer should be a numpy array of length 3
# make sure that the entries sums to 1!
#problem1_stationary = XXX

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
```

```python
# Part 5
#SOLUTION1
# Make Suburbs absorbing
P_abs = P.copy()
P_abs[1] = [0.0, 1.0, 0.0]

# Start in Downtown
v = np.array([1.0, 0.0, 0.0])

E_approx = 0.0
for t in range(30):                    # t = 0..29
    prob_survive = 1.0 - v[1]          # P(T > t)
    E_approx += prob_survive
    v = v @ P_abs                      # advance one step

print("Approx E[T] (t<30) =", E_approx)
# Fill in the answer to part 5 below
# That is, the expected number of steps
problem1_ET = E_approx

#SOLUTION2
"""
# Target is Suburbs (state 1). Solve for h(D), h(C) with h(S)=0.
# For i in {0,2}: h(i) = 1 + sum_j P[i,j] h(j), with h(1)=0.

A = np.array([
    [1 - P[0,0],    -P[0,2]],
    [   -P[2,0],  1 - P[2,2]]
], dtype=float)

b = np.array([1.0, 1.0])

hD, hC = np.linalg.solve(A, b)
print("Exact E[T | start=Downtown] =", hD)
"""
```

<a id="part-5"></a>

## Part 5

Double click this cell to enter edit mode and write your answer for part 5 below this line.

We consider the first hitting time of the suburbs. Let $h(i)$ denote the expected
number of steps to reach the suburbs starting from state $i$. Then
$h(\text{suburbs}) = 0$, and for all other states $i$ the expectations satisfy
the first-step equation

$$
h(i) = 1 + \sum_{j} P_{ij}\, h(j).
$$

Solving the resulting system of linear equations gives the expected hitting time.

---

<a id="examjanuary-2023-problem2-linear-regression"></a>
## ExamJanuary_2023.PROBLEM2 – Linear regression
Maximum Points = 13


You are given the "Abalone" dataset found in `data/abalone.csv`, which contains physical measurements of abalone (a type of sea shells) and the age of the abalone measured in **rings** (the number of rings in the shell) [https://en.wikipedia.org/wiki/Abalone](https://en.wikipedia.org/wiki/Abalone). Your task is to train a `linear regression` model to predict the age (Rings) of an abalone based on its physical measurements.

To evaluate your model, you will split the dataset into a training set and a testing set. You will use the training set to train your model, and the testing set to evaluate its performance.

1. Load the data into a pandas dataframe `problem2_df`. Based on the column names, figure out what are the features and the target and fill in the answer in the correct cell below. [2p]
2. Split the data into train and test. [2p]
3. Train the model. [1p]
4. On the test set, evaluate the model by computing the mean absolute error and plot the empirical distribution function of the residual with confidence bands (i.e. using the DKW inequality and 95% confidence). Hint: you can use the function `plotEDF,makeEDF` combo from `Utils.py` that we have used numerous times, which also contains the option to have confidence bands. [3p]
5. Provide a scatter plot where the x-axis corresponds to the predicted value and the y-axis is the true value, do this over the test set. [2p]
6. Reason about the performance, for instance, is the value of the mean absolute error good/bad and what do you think about the scatter plot in point 5? [3p]

```python
# Part 1
# Let problem2_df be the pandas dataframe that contains the data from the file
# data/abalone.csv
#problem2_df = XXX

import csv
import numpy as np
import pandas as pd

csv_path = "data/abalone.csv"

# Read the CSV file
df = pd.read_csv(csv_path)
problem2_df = df

# Display first few rows
print(problem2_df.head())

# Optional: display full DataFrame in Jupyter/Colab
#df
```

```python
# Part 1
# clean column names if they have BOM or extra whitespace
problem2_df.columns = problem2_df.columns.str.replace('\ufeff', '', regex=False).str.strip()
# Fill in the features as a list of strings of the names of the columns

problem2_features = [['Length', 'Diameter', 'Height','Whole weight','Shucked weight','Viscera weight', 'Shell weight']]
print("Features:", problem2_features)
# Fill in the target as a string with the correct column name

problem2_target = " Rings"
print("Target:", problem2_target)
```

```python
# Part 2


# Split the data into train and test using train_test_split
# keep the train size as 0.8 and use random_state=42
import numpy as np
from sklearn.model_selection import train_test_split
X = problem2_df[['Length', 'Diameter', 'Height','Whole weight','Shucked weight','Viscera weight', 'Shell weight']].to_numpy(dtype='float64')  
y =problem2_df['Rings'].to_numpy(dtype='int64')

#make split, keep size as 0.8
problem2_X_train,problem2_X_test,problem2_y_train,problem2_y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)

#display
print("Columns:",problem2_df.columns.tolist())
print("X", X)
print("Y:", y)
```

```python
# Part 3
# Include the necessary imports
from sklearn.linear_model import LinearRegression

# Initialize your linear regression model
problem2_model = LinearRegression()

# Train your model on the training data
problem2_model.fit(problem2_X_train, problem2_y_train)
```

```python
# Part 4

# Evaluate the model by computing the mean absolute error 
from sklearn.metrics import mean_absolute_error

# Predict on the test set
y_pred = problem2_model.predict(problem2_X_test)

# Evaluate the model by computing the mean absolute error
problem2_mae = mean_absolute_error(problem2_y_test, y_pred)

print('mae:', problem2_mae)
```

```python
# Part 4

# Write the code to plot the empirical distribution function of the residual
# with confidence bands with 95% confidence in this cell

# from Utils import makeEDF,plotEDF
# From Utils import makeEDF, plotEDF
from Utils import makeEDF, plotEDF

# Compute residuals
residuals = problem2_y_test - y_pred

# Create EDF object
edf = makeEDF(residuals)

# Plot EDF with 95% confidence bands (DKW inequality)
plotEDF(edf, alpha=0.05)
```

```python
# Part 5

# Write the code below to produce the scatter plot for part 5
import matplotlib.pyplot as plt
import numpy as np

plt.figure()
plt.scatter(y_pred, problem2_y_test)
plt.xlabel("Predicted value (y_pred)")
plt.ylabel("True value (y_test)")
plt.title("Predicted vs True (test set)")

# Optional: reference line y = x
mn = min(np.min(y_pred), np.min(problem2_y_test))
mx = max(np.max(y_pred), np.max(problem2_y_test))
plt.plot([mn, mx], [mn, mx])  # no color specified, default matplotlib color
plt.grid(True)
plt.show()
```

<a id="part-6"></a>

## Part 6

Double click this cell to enter edit mode and write your answer for part 6 below this line.

#### Discussion on the value of the MAE

The mean absolute error on the test set is MAE = 1.62. This means that, on average, the model’s predictions deviate from the true values by about 1.6 units. Given that the true values range roughly from about 5 to over 20, this error is relatively small compared to the overall scale of the response variable. Therefore, the MAE indicates a reasonably good predictive performance of the linear regression model.

#### Discussion on the predicted vs. true scatterplot

In the scatter plot of predicted versus true values, most points lie close to the diagonal line 
y=x, which indicates that the model predictions are generally accurate. The spread around the diagonal increases slightly for larger values, suggesting that prediction uncertainty grows for higher true values. There are no strong systematic deviations from the diagonal, which implies that the model does not exhibit a clear bias such as consistent over- or under-prediction.

#### Discussion

Overall, the model performs well on the test set. The relatively low MAE suggests good average accuracy, while the scatter plot shows a strong linear relationship between predicted and true values. However, the increasing spread for larger values and a few outliers indicate that the linear model may not fully capture all patterns in the data. This suggests that more complex models or additional features could potentially improve performance, especially for extreme values.

---

<a id="examjanuary-2023-problem3-count-regression-visits"></a>
## ExamJanuary_2023.PROBLEM3 – Count regression (visits)
Maximum Points = 13


A healthcare organization is interested in understanding the relationship between the number of visits to the doctors office and certain patient characteristics. 
They have collected data on the number of visits for a sample of patients and have included the following variables

* ofp : number of physician office visits
* ofnp : number of nonphysician office visits
* opp : number of physician outpatient visits
* opnp : number of nonphysician outpatient visits
* emr : number of emergency room visits
* hosp : number of hospitalizations
* exclhlth : the person is of excellent health (self-perceived)
* poorhealth : the person is of poor health (self-perceived)
* numchron : number of chronic conditions
* adldiff : the person has a condition that limits activities of daily living ?
* noreast : the person is from the north east region
* midwest : the person is from the midwest region
* west : the person is from the west region
* age : age in years (divided by 10)
* male : is the person male ?
* married : is the person married ?
* school : number of years of education
* faminc : family income in 10000$
* employed : is the person employed ?
* privins : is the person covered by private health insurance?
* medicaid : is the person covered by medicaid ?

Decide which patient features are resonable to use to predict the target "number of physician office visits". Hint: should we really use the "ofnp" etc variables?

Since the target variable is counts, a reasonable loss function is to consider the target variable as Poisson distributed where the parameter follows $\lambda = \exp(\alpha \cdot x + \beta)$ where $\alpha$ is a vector (slope) and $\beta$ is a number (intercept). That is, the parameter is the exponential of a linear function. The reason we chose this as our parameter, is that it is always positive which is when the Poisson distribution is defined. To be specific we make the following assumption about our conditional density of $Y \mid X$,
$$
    f_{Y \mid X} (y,x) = \frac{\lambda^{y} e^{-\lambda}}{y !}, \quad \lambda(x) = \exp(\alpha \cdot x + \beta).
$$

Recall from the lecture notes, (4.2) that in this case we should consider the log-loss (entropy) and that according to (4.2.1 Maximum Likelihood and regression) we can consider the conditional log-likelihood. Follow the steps of Example 1 and Example 2 in section (4.2) to derive the loss that needs to be minimized.

Hint: when taking the log of the conditional density you will find that the term that contains the $y!$ does not depend on $\lambda$ and as such does not depend on $\alpha,\beta$, it can thus be discarded. This will be essential due to numerical issues with factorials.

Instructions:

1. Load the file `data/visits_clean.csv` into the pandas dataframe `problem3_df`. Decide what should be features and target, give motivations for your choices. [3p]
2. Create the `problem3_X` and the `problem3_y` as numpy arrays with `problem3_X` being the features and `problem3_y` being the target. Do the standard train-test split with 80% training data and 20% testing data. Store these in the variables defined in the cells. [3p]
3. Implement $loss$ inside the class `PoissonRegression` by writing down the loss to be minimized, I have provided a formula for the $\lambda$ that you can use. [2p]
4. Now use the `PoissonRegression` class to train a Poisson regression model on the training data. [2p]
5. Come up with a reasonable metric to evaluate your model on the test data, compute it and write down a justification of this. Also, interpret your result and compare it to something naive. [3p]

```python
# Part 1

# Let problem3_df be the pandas dataframe that contains the data from the file
# data/visits_clean.csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1) Load data
#problem3_df = pd.read_csv("data/visits_clean.csv")
problem3_df = pd.read_csv(
    "data/visits_clean.csv",
    sep=r"\s+",
    engine="python"
)
print(problem3_df.columns.tolist())
```

```python
# Part 1
problem3_features = [
    "exclhlth", "poorhlth", "numchron", "adldiff",
    "noreast", "midwest", "west",
    "age", "male", "married", "school",
    "faminc", "employed", "privins", "medicaid"
]

problem3_target = "ofp"

#problem3_X = problem3_df[problem3_features].to_numpy(dtype=float)
#problem3_y = problem3_df[problem3_target].to_numpy(dtype=float)
```

<a id="part-1"></a>

## Part 1

Double click this cell to enter edit mode and write your answer for part 1 below this line.

#### What features are reasonable?
Use patient characteristics that are known before visits happen (demographics, health status, insurance, region).
Avoid variables that are basically other utilization counts because they are very close to the target and often measured in the same time window (high leakage / “cheating”).
#### In regards to how much data we have, how many features do you think we should aim for?
Rule of thumb: keep features modest relative to sample size and complexity. With typical datasets like this (often a few thousand rows), ~10–15 features is safe and interpretable. If you have very little data, reduce further.Reasonable predictors (non-leaky):
Health: excclth, poorhealth, numchron, alldiff
Demographics: age, male, married, school
Socio-economic: famin, employed
Insurance: privins, medicaid
Region dummies: noreast, midwest, west (baseline is “south/other”)
#### What other features would you like to have used but was not collected?
Examples:
distance/travel time to clinics, access to primary care
detailed diagnoses (ICD groups), severity scores, comorbidity index
prior-year utilization (but only if from an earlier period → no leakage)
appointment availability / waiting time
lifestyle factors (smoking, BMI), medication use
#### Discussion

```python
# Part 2

# Fill in your X and y below
#problem3_X = XXX
#problem3_y = XXX

# Split the data into train and test using train_test_split
# keep the train size as 0.8 and use random_state=42
#problem3_X_train, problem3_X_test, problem3_y_train, problem3_y_test = XXX

# Build X and y
problem3_X = problem3_df[problem3_features].to_numpy(dtype=float)
problem3_y = problem3_df[problem3_target].to_numpy(dtype=float)

# Train/test split
problem3_X_train, problem3_X_test, problem3_y_train, problem3_y_test = train_test_split(
    problem3_X, problem3_y, train_size=0.8, random_state=42
)

problem3_X_train.shape, problem3_X_test.shape, problem3_y_train.shape, problem3_y_test.shape
```

```python
# Part 3

# Fill in the function loss below

class PoissonRegression(object):
    def __init__(self):
        self.coeffs = None
        self.result = None
    
    def fit(self,X,Y):
        import numpy as np
        from scipy import optimize

        # define the objective/cost/loss function we want to minimise
        def loss(coeffs):
            # The parameter lambda for the given X and the proposed values 
            # of the coefficients, here coeff[:-1] represent alpha 
            # and coeff[-1] represent beta
            lam = np.exp(np.dot(X,coeffs[:-1])+coeffs[-1])

            # use the Y variable that is available here to define 
            # the loss function, return the value of the loss for 
            # this Y and for this parameter lam defined above
            return np.sum(lam - Y * np.log(lam))

        #Use the loss above together with an optimization method from scipy
        #to find the coefficients of the model
        #this is prepared for you below

        initial_arguments = np.zeros(shape=X.shape[1]+1) # initial guess as 0
        self.result = optimize.minimize(loss, initial_arguments,method='cg')
        self.coeffs = self.result.x
    
    def predict(self,X):
        #Use the trained model to predict Y
        if (self.coeffs is not None):
            return np.exp(np.dot(X,self.coeffs[:-1])+self.coeffs[-1])
```

```python


# Part 4

# Initialize your PoissonRegression model
problem3_model = PoissonRegression()

# Fit your initialized model on the training data
problem3_model.fit(problem3_X_train, problem3_y_train)

# This is to make sure that everything went well,
# check that success is True
print(problem3_model.result)
```

```python
# Part 5


import numpy as np

# Predict lambda on test set
lam_test = problem3_model.predict(problem3_X_test)

# Mean Poisson NLL (dropping log(y!) constant)
problem3_metric = np.mean(lam_test - problem3_y_test * np.log(lam_test))
problem3_metric

# Naive predictor: constant lambda = mean of training targets
lam_naive = np.mean(problem3_y_train) * np.ones_like(problem3_y_test)

naive_metric = np.mean(lam_naive - problem3_y_test * np.log(lam_naive))
naive_metric
```

<a id="part-5-2"></a>

## Part 5

Double click this cell to enter edit mode and write your answer for part 5 below this line.

#### Discussion on reasonable metrics and discussion about the value of the metric

#### Comparison with a naive model
**Metric choice:** Since the target is a count and we model \(Y|X\sim\text{Poisson}(\lambda)\) with
\(\lambda=\exp(\alpha^\top x+\beta)\), a natural evaluation metric is the Poisson negative log-likelihood
(ignoring the constant \(\log(y!)\) term). This metric matches the assumed probabilistic model and is the
same objective optimized during training.

**Interpreting the value:** Lower is better. The value should be compared to a naive baseline.

**Naive comparison:** A simple naive model is to predict a constant rate \(\lambda=\bar y_{\text{train}}\)
for all test points. If the learned model has lower test NLL than this baseline, the covariates provide
predictive power; otherwise the model is not improving beyond predicting the overall mean.

<a id="exam2024-problem1-random-variable-generation-rejection-inversion-sampling"></a>
## Exam2024.PROBLEM1 – Random variable generation: rejection & inversion sampling
Maximum Points = 14


In this problem you will do rejection sampling from complicated distributions, you will also be using your samples to compute certain integrals, a method known as Monte Carlo integration: (Keep in mind that choosing a good sampling distribution is often key to avoid too much rejection)

1. [4p] Fill in the remaining part of the function `problem1_inversion` in order to produce samples from the below distribution using rejection sampling:

$$
    F[x] =
    \begin{cases}
        0, & x \leq 0 \\
        \frac{e^{x^2}-1}{e-1}, & 0 < x < 1 \\
        1, & x \geq 1
    \end{cases}
$$

2. [2p] Produce 100000 samples (**use fewer if it times-out and you cannot find a solution**) and put the answer in `problem1_samples` from the above distribution and plot the histogram together with the true density. *(There is a timeout decorator on this function and if it takes more than 10 seconds to generate 100000 samples it will timeout and it will count as if you failed to generate.)*
3. [2p] Use the above 100000 samples (`problem1_samples`) to approximately compute the integral

$$
    \int_0^{1} \sin(x) \frac{2e^{x^2} x}{e-1} dx
$$
and store the result in `problem1_integral`.

4. [2p] Use Hoeffdings inequality to produce a 95\% confidence interval of the integral above and store the result as a tuple in the variable `problem1_interval`

5. [4p] Fill in the remaining part of the function `problem1_inversion_2` in order to produce samples from the below distribution using rejection sampling:
$$
    F[x] =
    \begin{cases}
        0, & x \leq 0 \\
        20xe^{20-1/x}, & 0 < x < \frac{1}{20} \\
        1, & x \geq \frac{1}{20}
    \end{cases}
$$
Hint: this is tricky because if you choose the wrong sampling distribution you reject at least 9 times out of 10. You will get points based on how long your code takes to create a certain number of samples, if you choose the correct sampling distribution you can easily create 100000 samples within 2 seconds.

```python
# Part 1
#from Utils import timeout
import numpy as np

#timeout
def problem1_inversion(n_samples=1):
    # Distribution from part 1
    # write the code in this function to produce samples from the distribution in the assignment
    # Make sure you choose a good sampling distribution to avoid unnecessary rejections
    u = np.random.rand(n_samples)  # U ~ Uniform(0,1)
    x = np.sqrt(np.log(1 + u * (np.e - 1)))

    # Return a numpy array of length n_samples
    return x
print(problem1_inversion(5))
```

```python
# Part 2
import numpy as np
import matplotlib.pyplot as plt

def problem1_samples():
    n = 100000

    # Inverse transform sampling
    u = np.random.rand(n)
    problem1_samples = np.sqrt(np.log(1 + u * (np.e - 1)))

    # Plot histogram
    x = np.linspace(0, 1, 500)
    true_density = 2 * x * np.exp(x**2) / (np.e - 1)

    plt.hist(problem1_samples, bins=100, density=True, alpha=0.6, label="Samples")
    plt.plot(x, true_density, 'r', lw=2, label="True density")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Histogram vs True Density")
    plt.show()

    return problem1_samples

#problem1_samples = XXX
```

```python
# Part 3
# --- Assume these are your actual integration limits and samples ---
a = 0.0  # Lower limit (example)
b = 1.0  # Upper limit (example)
# problem1_samples = np.random.rand(100000) # Your actual samples here
# For demonstration, let's use random samples from [0, 1]
problem1_samples = np.random.uniform(a, b, 100000)

# 1. Define function f(x) you want to integrate (replace with your actual function)
def f(x):
    # Example: Integrate x^2 from 0 to 1 (true answer is 1/3)
    return x**2

# 2. Evaluate the function at each sample point
function_values = f(problem1_samples)

# 3. Calculate the average of the function values
average_f = np.mean(function_values)

# 4. Apply the Monte Carlo formula: Integral ≈ (b - a) * average(f(x))
problem1_integral = (b - a) * average_f

# Print the result
print(f"Approximate integral: {problem1_integral}")
#problem1_integral = XXX
```

```python
# Part 4
import numpy as np

# 1) Generate samples (THIS WAS MISSING)
problem1_samples = problem1_inversion(100000)

# 2) Apply sin to the samples
Y = np.sin(problem1_samples)

# 3) Monte Carlo estimate of the integral
problem1_integral = np.mean(Y)

print("Integral estimate:", problem1_integral)
delta = 0.05
n = len(Y)

a = 0.0
b = np.sin(1.0)

epsilon = (b - a) * np.sqrt(np.log(2.0 / delta) / (2.0 * n))

problem1_interval = (
    problem1_integral - epsilon,
    problem1_integral + epsilon
)

print("95% Hoeffding CI:", problem1_interval)
```

```python
# Part 5
import numpy as np

def problem1_inversion_2(n_samples=1, batch_size=200_000):
    """
    Rejection sampling for X with CDF:
        F(x)=0 for x<=0
        F(x)=20 x exp(20 - 1/x) for 0<x<1/20
        F(x)=1 for x>=1/20

    Trick: sample Y = 1/X on [20, infinity) with a shifted Exp(1) proposal.
    Then return X = 1/Y.
    """

    # Bounding constant M = sup_{y>=20} 20(1/y + 1/y^2) = 1.05
    M = 1.05

    accepted = []  # will store accepted X values

    # Keep sampling until we have n_samples accepted values
    while True:
        need = n_samples - sum(len(a) for a in accepted)
        if need <= 0:
            break

        # Choose how many proposals to draw in this batch.
        # Since acceptance ~95%, drawing ~need/0.95 is enough.
        m = max(1000, int(need / 0.95) + 10)

        # 1) Sample proposal for Y: Y = 20 + Exp(1)
        y = 20.0 + np.random.exponential(scale=1.0, size=m)

        # 2) Compute acceptance probability:
        # f_Y(y)/g_Y(y) = 20(1/y + 1/y^2), so accept prob = ratio/M
        ratio = 20.0 * (1.0 / y + 1.0 / (y**2))
        accept_prob = ratio / M  # should be <= 1

        # 3) Accept/reject using uniform(0,1)
        u = np.random.rand(m)
        mask = (u <= accept_prob)

        # Convert accepted y to x = 1/y
        x_acc = 1.0 / y[mask]

        accepted.append(x_acc)

    # Concatenate and cut to exactly n_samples
    samples = np.concatenate(accepted)[:n_samples]
    return samples

#def problem1_inversion_2(n_samples=1):
    # Distribution from part 2
    # write the code in this function to produce samples from the distribution in the assignment
    # Make sure you choose a good sampling distribution to avoid unnecessary rejections

    # Return a numpy array of length n_samples
 #   return XXX
```

---
### Local test for Exam2024.PROBLEM1
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.

```python

# This cell is just to check that you got the correct formats of your answer
import numpy as np
try:
    assert(isinstance(problem1_inversion(10), np.ndarray))
except:
    print("Try again. You should return a numpy array from problem1_inversion")
else:
    print("Good, your problem1_inversion returns a numpy array")

try:
    assert(isinstance(problem1_samples, np.ndarray))
except:
    print("Try again. your problem1_samples is not a numpy array")
else:
    print("Good, your problem1_samples is a numpy array")

try:
    assert(isinstance(problem1_integral, float))
except:
    print("Try again. your problem1_integral is not a float")
else:
    print("Good, your problem1_integral is a float")

try:
    assert(isinstance(problem1_interval, list) or isinstance(problem1_interval, tuple)) , "problem1_interval not a tuple or list"
    assert(len(problem1_interval) == 2) , "problem1_interval does not have length 2, it should have a lower bound and an upper bound"
except Exception as e:
    print(e)
else:
    print("Good, your problem1_interval is a tuple or list of length 2")

try:
    assert(isinstance(problem1_inversion_2(10), np.ndarray))
except:
    print("Try again. You should return a numpy array from problem1_inversion_2")
else:
    print("Good, your problem1_inversion_2 returns a numpy array")
```

---

<a id="exam2024-problem2-logistic-regression-calibration-predictive-intervals-spam-example"></a>
## Exam2024.PROBLEM2 – Logistic regression: calibration & predictive intervals (spam example)
Maximum Points = 13


Let us build a proportional model ($\mathbb{P}(Y=1 \mid X) = G(\beta_0+\beta \cdot X)$ where $G$ is the logistic function) for the spam vs not spam data. Here we assume that the features are presence vs not presence of a word, let $X_1,X_2,X_3$ denote the presence (1) or absence (0) of the words $("free", "prize", "win")$.

1. [2p] Load the file `data/spam.csv` and create two numpy arrays, `problem2_X` which has shape (n_emails,3) where each feature in `problem2_X` corresponds to $X_1,X_2,X_3$ from above, `problem2_Y` which has shape **(n_emails,)** and consists of a $1$ if the email is spam and $0$ if it is not. Split this data into a train-calibration-test sets where we have the split $40\%$, $20\%$, $40\%$, put this data in the designated variables in the code cell.

2. [4p] Follow the calculation from the lecture notes where we derive the logistic regression and implement the final loss function inside the class `ProportionalSpam`. You can use the `Test` cell to check that it gives the correct value for a test-point.

3. [4p] Train the model `problem2_ps` on the training data. The goal is to calibrate the probabilities output from the model. Start by creating a new variable `problem2_X_pred` (shape `(n_samples,1)`) which consists of the predictions of `problem2_ps` on the calibration dataset. Then train a calibration model using `sklearn.tree.DecisionTreeRegressor`, store this trained model in `problem2_calibrator`.

4. [3p] Use the trained model `problem2_ps` and the calibrator `problem2_calibrator` to make final predictions on the testing data, store the prediction in `problem2_final_predictions`. Compute the $0-1$ test-loss and store it in `problem2_01_loss` and provide a $99\%$ confidence interval of it, store this in the variable `problem2_interval`, this should again be a tuple as in **problem1**.

```python
# Part 1
import numpy as np
import pandas as pd

#Assign path
spam_csv_path = "data/spam.csv"
#Read the CSV file into a DataFrame
#ver1 df_spam = pd.read_csv(spam_csv_path, header=None, encoding='utf-8', encoding_errors='ignore')
df_spam = pd.read_csv(spam_csv_path, encoding='latin-1')
df_spam.head()

#Create numpy arrays problem2_X and problem2_Y
# column names not known in advance
label_col = df_spam.columns[0]   # "v1"  -> ham/spam
text_col  = df_spam.columns[1]   # "v2"  -> message text

# Build problem2_X with features for X1="free", X2="prize", X3="win"
words = ["free", "prize", "win"]

# Number of texts in the dataset and number of words to search for
n_texts = len(df_spam) # number of rows in the DataFrame
problem2_X = np.zeros((n_texts, len(words)), dtype=int) # initialize feature matrix

# Fill the feature matrix
for j, w in enumerate(words):
    problem2_X[:, j] = df_spam[text_col].str.contains(    # search for whole word w
        fr"\b{w}\b", case=False, na=False                   # regex=True
    ).astype(int)     # convert boolean to int (0/1)

# Build problem3_Y as 1 for "spam" and 0 for "ham"
problem2_Y = (df_spam[label_col].str.lower() == "spam").astype(int).to_numpy()

# Split the data into training (40%), calibration (20%), and test (40%) sets
n_train = int(0.4 * n_texts)
n_calib = int(0.2 * n_texts)
n_test = n_texts - n_train - n_calib

problem2_X_train = problem2_X[:n_train]
problem2_X_calib = problem2_X[n_train:n_train + n_calib]
problem2_X_test = problem2_X[n_train + n_calib:]
problem2_Y_train = problem2_Y[:n_train]
problem2_Y_calib = problem2_Y[n_train:n_train + n_calib]
problem2_Y_test = problem2_Y[n_train + n_calib:]
print(problem2_X_train.shape,problem2_X_calib.shape,problem2_X_test.shape,problem2_Y_train.shape,problem2_Y_calib.shape,problem2_Y_test.shape)
```

```python
# Part 2



class ProportionalSpam(object):
    def __init__(self):
        self.coeffs = None
        self.result = None
    
    # define the objective/cost/loss function we want to minimise
    def loss(self,X,Y,coeffs):
            import numpy as np

            # linear part  z_i = w0 + x_i·w
            z = np.dot(X, coeffs[1:]) + coeffs[0]

            # logistic function σ(z) = 1 / (1 + e^(−z))
            p = 1.0 / (1.0 + np.exp(-z))   # predicted probability P(y=1 | x)

            # negative log-likelihood / cross-entropy loss
            eps = 1e-15                    # to avoid log(0)
            p = np.clip(p, eps, 1 - eps)   # keep p in (0,1)
            loss = -np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p))

            return loss

    def fit(self,X,Y):
        import numpy as np
        from scipy import optimize

        #Use the f above together with an optimization method from scipy
        #to find the coefficients of the model
        opt_loss = lambda coeffs: self.loss(X,Y,coeffs)
        initial_arguments = np.zeros(shape=X.shape[1]+1)
        self.result = optimize.minimize(opt_loss, initial_arguments,method='cg')
        self.coeffs = self.result.x
    
    def predict(self,X):
        #Use the trained model to predict Y
        if (self.coeffs is not None):
            G = lambda x: np.exp(x)/(1+np.exp(x))
            return np.round(10*G(np.dot(X,self.coeffs[1:])+self.coeffs[0]))/10 # This rounding is to help you with the calibration
```

```python
# Part 3
# 1. Train ProportionalSpam on the training data
problem2_ps = ProportionalSpam()
problem2_ps.fit(problem2_X_train, problem2_Y_train)

# 2. Get predicted probabilities on the *calibration* set
#    problem2_X_pred must have shape (n_samples, 1)
raw_calib_pred = problem2_ps.predict(problem2_X_calib)   # shape (n_samples,)
problem2_X_pred = raw_calib_pred.reshape(-1, 1)        # shape (n_samples, 1)

# 3. Train a calibration model (DecisionTreeRegressor)
from sklearn.tree import DecisionTreeRegressor

problem2_calibrator = DecisionTreeRegressor(
    max_depth=3,
    random_state=0
)
problem2_calibrator.fit(problem2_X_pred, problem2_Y_calib)


print("problem2_X_pred", problem2_X_pred.shape)
print("problem2_calibrator", problem2_calibrator)
#problem2_ps = XXX

#problem2_X_pred = XXX

#problem2_calibrator = XXX
```

```python
# Part 4
import numpy as np

# ---------- 1) Get uncalibrated probabilities from the trained model ----------
# If your model has predict_proba, use it; otherwise use your model's probability output method.
if hasattr(problem2_ps, "predict_proba"):
    # sklearn-style: returns shape (n,2) for classes [0,1]
    p_uncal = problem2_ps.predict_proba(problem2_X_test)[:, 1]
else:
    # custom model might return probabilities directly
    # change "predict" to whatever your model uses for prob output
    p_uncal = problem2_ps.predict(problem2_X_test)

# Ensure correct shape for the calibrator: (n_samples, 1)
problem2_X_test_pred = p_uncal.reshape(-1, 1)

# ---------- 2) Apply the trained calibrator to get calibrated probabilities ----------
# DecisionTreeRegressor outputs a real number; clip to [0,1] for safety.
p_cal = problem2_calibrator.predict(problem2_X_test_pred)
p_cal = np.clip(p_cal, 0.0, 1.0)

# Store as required
problem2_final_predictions = p_cal  # shape (n_test,)

# ---------- 3) Convert probabilities to final 0/1 class predictions ----------
# Standard threshold is 0.5 unless the exam states otherwise.
y_hat = (problem2_final_predictions >= 0.5).astype(int)

# ---------- 4) Compute 0–1 loss on the test set ----------
# 0–1 loss = fraction of incorrect classifications.
problem2_01_loss = np.mean(y_hat != problem2_Y_test)

# ---------- 5) 99% Hoeffding confidence interval for the 0–1 loss ----------
# Define error indicators E_i = 1[y_hat_i != y_i], bounded in [0,1]
# Then Hoeffding: P(|mean(E)-E[E]| >= eps) <= 2 exp(-2n eps^2)
# For confidence 1-delta = 0.99 => delta = 0.01
delta = 0.01
n = len(problem2_Y_test)

# Since E_i in [0,1], (b-a)=1
epsilon = np.sqrt(np.log(2.0 / delta) / (2.0 * n))

lower = max(0.0, problem2_01_loss - epsilon)
upper = min(1.0, problem2_01_loss + epsilon)

problem2_interval = (lower, upper)

# Optional prints 
print("0-1 test loss:", problem2_01_loss)
print("99% Hoeffding CI:", problem2_interval)
```

---
### Local test for Exam2024.PROBLEM2
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.

```python
try:
    import numpy as np
    test_instance = ProportionalSpam()
    test_loss = test_instance.loss(np.array([[1,0,1],[0,1,1]]),np.array([1,0]),np.array([1.2,0.4,0.3,0.9]))
    assert (np.abs(test_loss-1.2828629432232497) < 1e-6)
    print("Your loss was correct for a test point")
except:
    print("Your loss was not correct on a test point")
```

---

<a id="exam2024-problem3-markov-chains-classification-of-properties-stationary-behavior"></a>
## Exam2024.PROBLEM3 – Markov chains: classification of properties & stationary behavior
Maximum Points = 13


Consider the following four Markov chains, answer each question for all chains:

<img width="400px" src="pictures/MarkovA.png">Markov chain A</img>
<img width="400px" src="pictures/MarkovB.png">Markov chain B</img>
<img width="400px" src="pictures/MarkovC.png">Markov chain C</img>
<img width="400px" src="pictures/MarkovD.png">Markov chain D</img>

1. [2p] What is the transition matrix?
2. [2p] Is the Markov chain irreducible?
3. [3p] Is the Markov chain aperiodic? What is the period for each state?
4. [3p] Does the Markov chain have a stationary distribution, and if so, what is it?
5. [3p] Is the Markov chain reversible?

```python
# PART 1

#------------------------TRANSITION MATRIX -------------------------------
# Answer each one by supplying the transition matrix as a numpy array
# of shape (n_states,n_states), where state (A,B,...) corresponds to index (0,1,...)
import numpy as np
problem3_A    = np.array([
    [0.8, 0.2, 0, 0],
    [0.6, 0.2, 0.2, 0],
    [0, 0.4, 0, 0.6],
    [0, 0, 0.8, 0.2]
])
problem3_B    = np.array([
    [0, 0.2, 0, 0.8],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0.5, 0, 0.5, 0]
])
problem3_C    = np.array([
    [0.2, 0.3, 0, 0, 0.5],
    [0.2, 0.2, 0.6, 0, 0],
    [0, 0.4, 0, 0.6, 0],
    [0, 0, 0, 0.6, 0.4],
    [0, 0, 0, 0.4, 0.6]
])
problem3_D    = np.array([
    [0.8, 0.2, 0, 0],
    [0.6, 0.2, 0.2, 0],
    [0, 0.4, 0, 0.6],
    [0.1, 0, 0.7, 0.2]
])
```

```python
# PART 2
#------------------------REDUCIBLE -------------------------------
# Answer each one with a True or False

problem3_A_irreducible = True
problem3_B_irreducible = False
problem3_C_irreducible = False
problem3_D_irreducible = True
```

```python
# PART 3
#------------------------APERIODIC-------------------------------
# Answer each one with a True or False

problem3_A_is_aperiodic = True
problem3_B_is_aperiodic = False
problem3_C_is_aperiodic = True
problem3_D_is_aperiodic = True

# Answer the following with the period of the states as a numpy array
# of shape (n_states,)

problem3_A_periods = ([1,1,1,1])
problem3_B_periods = ([2,2,2,2])
problem3_C_periods = ([1,1,1,1,1])
problem3_D_periods = ([1,1,1,1])
```

```python
# PART 4
#------------------------STATIONARY DISTRIBUTION-----------------
# Answer each one with a True or False

problem3_A_has_stationary = True
problem3_B_has_stationary = True
problem3_C_has_stationary = True
problem3_D_has_stationary = True

# Answer the following with the stationary distribution as a numpy array of shape (n_states,)
# if the Markov chain has a stationary distribution otherwise answer with False

def stationary_distribution(P):
    P = np.asarray(P, dtype=float)
    n = P.shape[0]

    A = P.T - np.eye(n)
    b = np.zeros(n)

    A[-1, :] = 1.0   # normalization
    b[-1] = 1.0

    pi = np.linalg.solve(A, b)
    pi = np.clip(pi, 0.0, 1.0)
    pi = pi / pi.sum()
    return pi

pi_ex_A = stationary_distribution(problem3_A)
pi_ex_B = stationary_distribution(problem3_B)
pi_ex_C = stationary_distribution(problem3_C)
pi_ex_D = stationary_distribution(problem3_D)
#print("Stationary pi:", pi_ex, "sum=", pi_ex.sum())

problem3_A_stationary_dist = pi_ex_A
problem3_B_stationary_dist = pi_ex_B
problem3_C_stationary_dist = pi_ex_C
problem3_D_stationary_dist = pi_ex_D
```

```python
# PART 5
#------------------------REVERSIBLE-----------------
# Answer each one with a True or False
def is_reversible(P, pi=None, tol=1e-8):
    # Check detailed balance pi_i P_ij == pi_j P_ji.
    P = np.asarray(P, dtype=float)
    if pi is None:
        pi = stationary_distribution(P)

    lhs = pi.reshape(-1, 1) * P
    rhs = (pi.reshape(1, -1) * P.T)
    return bool(np.allclose(lhs, rhs, atol=tol, rtol=0))

problem3_A_is_reversible = is_reversible(problem3_A, pi_ex_A)
problem3_B_is_reversible = is_reversible(problem3_B, pi_ex_B)
problem3_C_is_reversible = is_reversible(problem3_C, pi_ex_C)
problem3_D_is_reversible = is_reversible(problem3_D, pi_ex_D)
#print(problem3_A_is_reversible)
```

---

<a id="examjanuary-2022-source-notebook"></a>
## ExamJanuary_2022 (source notebook)

# Exam 2021, 8.00-13.00 for the course 1MS041 (Introduction to Data Science / Introduktion till dataanalys)

## Instructions:
1. Complete the problems by following instructions.
2. When done, submit this file with your solutions saved, following the instruction sheet.

This exam has 3 problems for a total of 40 points, to pass you need
20 points.

## Some general hints and information:
* Try to answer all questions even if you are uncertain.
* Comment your code, so that if you get the wrong answer I can understand how you thought
this can give you some points even though the code does not run.
* Follow the instruction sheet rigorously.
* This exam is partially autograded, but your code and your free text answers are manually graded anonymously.
* If there are any questions, please ask the exam guards, they will escalate it to me if necessary.
* I (Benny) will visit the exam room at around 10:30 to see if there are any questions.

## Tips for free text answers
* Be VERY clear with your reasoning, there should be zero ambiguity in what you are referring to.
* If you want to include math, you can write LaTeX in the Markdown cells, for instance `$f(x)=x^2$` will be rendered as $f(x)=x^2$ and `$$f(x) = x^2$$` will become an equation line, as follows
$$f(x) = x^2$$
Another example is `$$f_{Y \mid X}(y,x) = P(Y = y \mid X = x) = \exp(\alpha \cdot x + \beta)$$` which renders as
$$f_{Y \mid X}(y,x) = P(Y = y \mid X = x) = \exp(\alpha \cdot x + \beta)$$

## Finally some rules:
* You may not communicate with others during the exam, for example:
    * You cannot ask for help in Stack-Overflow or other such help forums during the Exam.
    * You may not communicate with AI's, for instance ChatGPT.
    * Your on-line and off-line activity is being monitored according to the examination rules.

## Good luck!

```python
# Insert your anonymous exam ID as a string in the variable below
examID="XXX"
```

---
## Exam vB, PROBLEM 1
Maximum Points = 8

---

<a id="examjanuary-2022-problem1-probability-warmup"></a>
## ExamJanuary_2022.PROBLEM1 – Probability warmup
## Probability warmup
Let's say we have an exam question which consists of $20$ yes/no questions.
From past performance of similar students, a randomly chosen student will know the correct answer to $N \sim \text{binom}(20,11/20)$ questions. Furthermore, we assume that the student will guess the answer with equal probability to each question they don't know the answer to, i.e. given $N$ we define $Z \sim \text{binom}(20-N,1/2)$ as the number of correctly guessed answers. Define $Y = N + Z$, i.e., $Y$ represents the number of total correct answers.

We are interested in setting a deterministic threshold $T$, i.e., we would pass a student at threshold $T$ if $Y \geq T$. Here $T \in \{0,1,2,\ldots,20\}$.

1. [5p] For each threshold $T$, compute the probability that the student *knows* less than $10$ correct answers given that the student passed, i.e., $N < 10$. Put the answer in `problem11_probabilities` as a list.
2. [3p] What is the smallest value of $T$ such that if $Y \geq T$ then we are 90\% certain that $N \geq 10$?

```python

# Hint the PMF of N is p_N(k) where p_N is
p = 11/20
p_N = lambda k: binomial(20,k)*(1-p)^(20-k)*(p)^k
```

```python

# Part 1:
# replace XXX to represent P(N < 10) for T = [0,1,2,...,20], i.e. your answer should be a list
# of length 21.
# Hint the PMF of N is p_N(k) where p_N is

#import scipy
from math import comb
#from scipy.special import binom as binomial

n = 20                  # number of questions
p = 11/20                # probability student knows a question
p_guess = 1/2          # probability of guessing correctly
p_N = lambda k: comb(20,k)*((1-p)**(20-k))*((p)**k)
#p_N = lambda k: binomial(20,k)*(1-p)^(20-k)*(p)^k
problem11_probabilities1 = []

for T in range(n+1):
    numerator = 0.0   # P(N<10 and Y>=T)
    denominator = 0.0 # P(Y>=T)

    for N in range(n+1):
        # Distribution of Z given N
        for z in range(n-N+1):
            prob = p_N(N) * comb(n-N, z) * (p_guess**z) * ((1-p_guess)**((n-N)-z))
            Y = N + z
            if Y >= T:
                denominator += prob
                if N < 10:
                    numerator += prob

    cond_prob = numerator / denominator if denominator > 0 else 0
    problem11_probabilities1.append(cond_prob)

for T, prob in enumerate(problem11_probabilities1):
    print(f"T={T}: P(N<10 | Y>=T) = {prob:.6f}")
#problem11_probabilities = [XXX,XXX,...,XXX]
```

```python

# Part 2: Give an integer between 0 and 20 which is the answer to 2.
problem12_T =  [0.102270]
```

---
## Exam vB, PROBLEM 2
Maximum Points = 8

---

<a id="examjanuary-2022-problem2-random-variable-generation-and-transformation"></a>
## ExamJanuary_2022.PROBLEM2 – Random variable generation and transformation
## Random variable generation and transformation

The purpose of this problem is to show that you can implement your own sampler, this will be built in the following three steps:

1. [2p] Implement a Linear Congruential Generator where you tested out a good combination (a large $M$ with $a,b$ satisfying the Hull-Dobell (Thm 6.8)) of parameters. Follow the instructions in the code block.
2. [2p] Using a generator construct random numbers from the uniform $[0,1]$ distribution.
3. [4p] Using a uniform $[0,1]$ random generator, generate samples from

$$p_0(x) = \frac{\pi}{2}|\sin(2\pi x)|, \quad x \in [0,1] \enspace .$$

Using the **Accept-Reject** sampler (**Algorithm 1** in TFDS notes) with sampling density given by the uniform $[0,1]$ distribution.

```python
import numpy as np

def problem2_LCG(size=None, seed = 0):
    """
    A linear congruential generator that generates pseudo random numbers according to size.

    Parameters
    -------------
    size : an integer denoting how many samples should be produced
    seed : the starting point of the LCG, i.e. u0 in the notes.

    Returns
    -------------
      out : a list of the pseudo random numbers
    """

    # Parameters satisfying Hull-Dobell theorem
    m = 2**48
    a = 25214903917
    c = 11

    # Initialize
    x = seed
    out = []

    # Generate pseudo-random numbers
    for _ in range(size):
        x = (a * x + c) % m
        out.append(x / m)  # Normalize to [0,1)

    return out
print(problem2_LCG(size=10, seed=0))  # example usage
```

```python

def problem2_uniform(generator=None, period = 1, size=None, seed=0):
    """
    Takes a generator and produces samples from the uniform [0,1] distribution according
    to size.

    Parameters
    -------------
    generator : a function of type generator(size,seed) and produces the same result as problem2_LCG, i.e. pseudo random numbers in the range {0,1,...,period-1}
    period : the period of the generator
    seed : the seed to be used in the generator provided
    size : an integer denoting how many samples should be produced

    Returns
    --------------
    out : a list of the uniform pseudo random numbers
    """
    if size is None:
        raise ValueError("size must be provided")
    if generator is None:
        raise ValueError("You must provide a generator function (e.g. problem1_LCG)")

    # Step 1: Generate pseudo-random integers using the provided generator
    random_integers = generator(size, seed)

    # Step 2: Convert to uniform [0,1) values
    uniform_numbers = [x / float(period) for x in random_integers]

    return uniform_numbers
print(problem2_uniform(generator=problem2_LCG, period=2**48, size=10, seed=0))  # example usage
    #XXX

    #return XXX
```

```python

def problem2_accept_reject(uniformGenerator=None, n_iterations=None, seed=0):
    """
    Takes a generator that produces uniform pseudo random [0,1] numbers
    and produces samples from (pi/2)*abs(sin(x*2*pi)) using an Accept-Reject
    sampler with the uniform distribution as the proposal distribution

    Parameters
    -------------
    generator : a function of the type generator(size,seed) that produces uniform pseudo random
    numbers from [0,1]
    seed : the seed to be used in the generator provided
    size : an integer denoting how many samples should be produced

    Returns
    --------------
    out : a list of the pseudo random numbers with the specified distribution
    """
    if n_iterations is None:
        raise ValueError("n_iterations must be provided")

    # Default Uniform(0,1) generator if none was passed
    if uniformGenerator is None:
        def uniformGenerator(size, seed):
            rng = np.random.default_rng(seed)
            return rng.random(size).tolist()

    # We need two uniforms per attempt: one for x, one for the acceptance test
    u = uniformGenerator(2 * n_iterations, seed)
    out = []
    idx = 0

    # M = sup_x p0(x)/q(x) with q(x)=1 on [0,1]. Since max |sin(2πx)| = 1, M = π/2.
    # Accept if U <= p0(x) / (M*q(x)) = |sin(2πx)|.
    for _ in range(n_iterations):
        x = u[idx]; idx += 1           # proposal from Uniform(0,1)
        u_acc = u[idx]; idx += 1       # uniform for acceptance test
        if u_acc <= abs(np.sin(2 * np.pi * x)):
            out.append(x)

    return out

    #XXX

    #return XXX
```

---
#### Local Test for Exam vB, PROBLEM 2
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.

```python

# If you managed to solve all three parts you can test the following code to see if it runs
# you have to change the period to match your LCG though, this is marked as XXX.
# It is a very good idea to check these things using the histogram function in sagemath
# try with a larger number of samples, up to 10000 should run

print("LCG output: %s" % problem2_LCG(size=10, seed = 1))

period = 1

print("Uniform sampler %s" % problem2_uniform(generator=problem2_LCG, period = period, size=10, seed=1))

uniform_sampler = lambda size,seed: problem2_uniform(generator=problem2_LCG, period = period, size=size, seed=seed)

print("Accept-Reject sampler %s" % problem2_accept_reject(uniformGenerator = uniform_sampler,n_iterations=20,seed=1))
```

```python

# If however you did not manage to implement either part 1 or part 2 but still want to check part 3, you can run the code below

def testUniformGenerator(size,seed):
    #set_random_seed(seed)
    rng(seed)

    return [random() for s in range(size)]

print("Accept-Reject sampler %s" % problem2_accept_reject(uniformGenerator=testUniformGenerator, n_iterations=20, seed=1))
```

---
## Exam vB, PROBLEM 3
Maximum Points = 8

---

<a id="examjanuary-2022-problem3-concentration-of-measure"></a>
## ExamJanuary_2022.PROBLEM3 – Concentration of measure
## Concentration of measure

As you recall, we said that concentration of measure was simply the phenomenon where we expect that the probability of a large deviation of some quantity becoming smaller as we observe more samples: [0.4 points per correct answer]

1. Which of the following will exponentially concentrate, i.e. for some $C_1,C_2,C_3,C_4 $
$$
    P(Z - \mathbb{E}[Z] \geq \epsilon) \leq C_1 e^{-C_2 n \epsilon^2} \wedge C_3 e^{-C_4 n (\epsilon+1)} \enspace .
$$

    1. The empirical mean of i.i.d. sub-Gaussian random variables?
    2. The empirical mean of i.i.d. sub-Exponential random variables?
    3. The empirical mean of i.i.d. random variables with finite variance?
    4. The empirical variance of i.i.d. random variables with finite variance?
    5. The empirical variance of i.i.d. sub-Gaussian random variables?
    6. The empirical variance of i.i.d. sub-Exponential random variables?
    7. The empirical third moment of i.i.d. sub-Gaussian random variables?
    8. The empirical fourth moment of i.i.d. sub-Gaussian random variables?
    9. The empirical mean of i.i.d. deterministic random variables?
    10. The empirical tenth moment of i.i.d. Bernoulli random variables?

2. Which of the above will concentrate in the weaker sense, that for some $C_1$
$$
    P(Z - \mathbb{E}[Z] \geq \epsilon) \leq \frac{C_1}{n \epsilon^2}?
$$

```python

# Answers to part 1, which of the alternatives exponentially concentrate, answer as a list
# i.e. [1,4,5] that is example 1, 4, and 5 concentrate
problem3_answer_1 = [2,4,5,8,9,10]
```

```python

# Answers to part 2, which of the alternatives concentrate in the weaker sense, answer as a list
# i.e. [1,4,5] that is example 1, 4, and 5 concentrate
problem3_answer_2 = [2,3,4,5,6,7,8,9,10]
```

---
## Exam vB, PROBLEM 4
Maximum Points = 8

---

<a id="examjanuary-2022-problem4-sms-spam-filtering"></a>
## ExamJanuary_2022.PROBLEM4 – SMS spam filtering
## SMS spam filtering [8p]

In the following problem we will explore SMS spam texts. The dataset is the `SMS Spam Collection Dataset` and we have provided for you a way to load the data. If you run the appropriate cell below, the result will be in the `spam_no_spam` variable. The result is a `list` of `tuples` with the first position in the tuple being the SMS text and the second being a flag `0 = not spam` and `1 = spam`.

1. [3p] Let $X$ be the random variable that represents each SMS text (an entry in the list), and let $Y$ represent whether text is spam or not i.e. $Y \in \{0,1\}$. Thus $\mathbb{P}(Y = 1)$ is the probability that we get a spam. The goal is to estimate:
$$
    \mathbb{P}(Y = 1 | \text{"free" or "prize" is in } X) \enspace .
$$
That is, the probability that the SMS is spam given that "free" or "prize" occurs in the SMS.
Hint: it is good to remove the upper/lower case of words so that we can also find "Free" and "Prize"; this can be done with `text.lower()` if `text` a string.

2. [3p] Provide a "90\%" interval of confidence around the true probability. I.e. use the Hoeffding inequality to obtain for your estimate $\hat P$ of the above quantity. Find $l > 0$ such that the following holds:
$$
    \mathbb{P}(\hat P - l \leq \mathbb{E}[\hat P] \leq \hat P + l) \geq 0.9 \enspace .
$$
3. [2p] Repeat the two exercises above for "free" appearing twice in the SMS.

```python

# Run this cell to get the SMS text data
from Utils import load_sms
#from exam_extras import load_sms
spam_no_spam = load_sms()
print(spam_no_spam)

#Typical structure - just for general undestanding of sms data
#spam_no_spam = [
#    ("Free prize now!!!", 1),
#    ("Hey are we meeting today?", 0),
#    ...
#]
```

```python

# fill in the estimate for part 1 here (should be a number between 0 and 1)

num_contains = 0
num_spam_and_contains = 0

for text, label in spam_no_spam:
    text_lower = text.lower()
    
    if ("free" in text_lower) or ("prize" in text_lower):
        num_contains += 1
        if label == 1:
            num_spam_and_contains += 1

problem4_hatP = num_spam_and_contains / num_contains
print(problem4_hatP)


#problem4_hatP = XXX
```

```python

# fill in the calculated l from part 2 here
import numpy as np


"""
def hoeffding_conf_interval(data=spam_no_spam, a=0, b=1, conf_level=0.90):
    alpha = 1 - conf_level
    n = len(data)
    epsilon = (b - a) / np.sqrt(2 * n) * np.sqrt(np.log(2 / alpha))
    mean = np.mean(data)
    return (mean - epsilon, mean + epsilon)
print("90% Hoeffding confidence interval:", hoeffding_conf_interval)
"""
n=num_contains
conf_level = 0.9
alpha = 1 - conf_level

problem4_l = np.sqrt((1 / (2 * n)) * np.log(2 / alpha))

lower = problem4_hatP - problem4_l
upper = problem4_hatP + problem4_l

print("90% confidence interval:", (lower, upper))
print("problem4_l:", problem4_l)
```

```python

# fill in the estimate for hatP for the double free question in part 3 here (should be a number between 0 and 1)
num_contains = 0
num_spam_and_contains = 0

for text, label in spam_no_spam:
    text_lower = text.lower()
    
    if text_lower.count("free") >= 2:
        num_contains += 1
        if label == 1:
            num_spam_and_contains += 1

problem4_hatP2 = num_spam_and_contains / num_contains
print(problem4_hatP2)
#problem4_hatP2 = XXX
```

```python

# fill in the estimate for l for the double free question in part 3 here
n=num_contains
conf_level = 0.9
alpha = 1 - conf_level

problem4_l2 = np.sqrt((1 / (2 * n)) * np.log(2 / alpha))

lower = problem4_hatP2 - problem4_l2
upper = problem4_hatP2 + problem4_l2

print("90% confidence interval:", (lower, upper))
print("problem4_l:", problem4_l2)
#problem4_l2 = XXX
```

---
## Exam vB, PROBLEM 5
Maximum Points = 8

---

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

```python
import pandas as pd
#read the fole
df = pd.read_csv("data/flights.csv")
#fill value of variables
number_of_observations = df.shape[0]
number_of_userCodes = df["userCode"].nunique()
number_of_cities = len(pd.unique(pd.concat([df["from"], df["to"]])))
#display variables 
number_of_cities, number_of_userCodes, number_of_observations
```

```python

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
```

```python

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
```

```python

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
```

```python

# Compute the return probability for part 3 of problem 5
# Compute tree-step transition matrix
i = cityToIndex['Aracaju (SE)']
transition_matrix_3 = np.linalg.matrix_power(transition_matrix, 3)
print(transition_matrix_3)
return_probability = transition_matrix_3[i, i]
print(return_probability_problem5)
```

---
#### Local Test for Exam vB, PROBLEM 5
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.

```python
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

---
## Exam vB, PROBLEM 6
Maximum Points = 8

---

<a id="examjanuary-2022-problem6-black-box-testing"></a>
## ExamJanuary_2022.PROBLEM6 – Black box testing
## Black box testing

In the following problem we will continue with our SMS spam / nospam data. This time we will try to approach the problem as a pattern recognition problem. For this particular problem I have provided you with everything -- data is prepared, split into train-test sets and a black-box model has been fitted on the training data and predicted on the test data. Your goal is to calculate test metrics and provide guarantees for each metric.

1. [2p] Compute precision for class 1 (see notes 8.3.2 for definition), then provide an interval using Hoeffding's inequality for a 95\% confidence.
2. [2p] Compute recall for class 1(see notes 8.3.2 for definition), then provide an interval using Hoeffding's inequality for a 95\% interval.
3. [2p] Compute accuracy (0-1 loss), then provide an interval using Hoeffding's inequality for a 95\% interval.
4. [2p] If we would have used a classifier with VC-dimension 3, would we have obtained a smaller interval for accuracy by using all data?

```python

# The code below will load data, split the data into train and test and run a "black box" algorithm on it
# the result of the "black box" is stored in predictions_problem6, the true values will be stored in
# Y_test_problem6

from exam_extras import load_sms_problem6
X_problem6, Y_problem6 = load_sms_problem6()

X_train_problem6,X_test_problem6,Y_train_problem6,Y_test_problem6 = exam_extras.train_test_split(X_problem6,Y_problem6)
predictions_problem6 = exam_extras.knn_predictions(X_train_problem6,Y_train_problem6,X_test_problem6,k=4)
```

```python

# Compute the precision of predictions_problem6 with respect to Y_test_problem6
#problem6_precision = XXX
import numpy as np

y_true = np.array(Y_test_problem6)
y_pred = np.array(predictions_problem6)

TP = np.sum((y_pred == 1) & (y_true == 1))
FP = np.sum((y_pred == 1) & (y_true == 0))

den_prec = TP + FP
problem6_precision = TP / den_prec if den_prec > 0 else 0.0
```

```python

# Compute the interval length l of precision of predictions_problem6 with respect to Y_test_problem6, with the same definition of l as in problem 4
# Hoeffding half-width l for 95% confidence, data in [0,1]
alpha = 0.05
problem6_precision_l = np.sqrt(np.log(2/alpha) / (2 * den_prec)) if den_prec > 0 else 0.0
#problem6_precision_l = XXX
```

```python

# Repeat the same procedure but for recall
#problem6_recall = XXX
FN = np.sum((y_pred == 0) & (y_true == 1))
den_rec = TP + FN
problem6_recall = TP / den_rec if den_rec > 0 else 0.0
```

```python
problem6_recall_l = np.sqrt(np.log(2/alpha) / (2 * den_rec)) if den_rec > 0 else 0.0
#problem6_recall_l = XXX
```

```python

# Repeat the same procedure but for accuracy or 0-1 loss
#problem6_accuracy = XXX
n_test = len(y_true)
problem6_accuracy = np.mean(y_pred == y_true)
```

```python


problem6_accuracy_l = np.sqrt(np.log(2/alpha) / (2 * n_test)) if n_test > 0 else 0.0

#problem6_accuracy_l = XXX
```

```python

# Below you will calculate the interval parameter l for a classifier running on all data with a VC dimension of 3
# put the value in problem6_VC_l and answer problem_VC_smaller as True if the interval is smaller than the test-accuracy above
# if not answer False. Make sure you replace XXX with something even if you only answer one of them.
#problem6_VC_l = XXX # number
#problem6_VC_smaller = XXX #True / False
d = 3
delta = 0.05

n_all = len(Y_problem6)  # full dataset labels loaded earlier

# VC half-width (one standard bound using Sauer + VC inequality)
term = 4 * ((2 * np.e * n_all / d) ** d) / delta
problem6_VC_l = np.sqrt((8 / n_all) * np.log(term)) if n_all > 0 else 0.0

# Compare to test accuracy interval half-width
problem6_VC_smaller = problem6_VC_l < problem6_accuracy_l
```

<a id="python-reference"></a>
# Python Reference (functions/commands used in solutions)

[Back to Index](#index-problem-theme-oriented)

---
## 1) Python built-ins

<a id="pyref-print"></a>
### `print(*objects, sep=' ', end='\n')`
**Used in:** [Assignment 1, PROBLEM 1](#assignment-1-problem-1), [Assignment 1, PROBLEM 2](#assignment-1-problem-2), [Assignment 1, PROBLEM 3](#assignment-1-problem-3), [Assignment 2, PROBLEM 1](#assignment-2-problem-1), [Assignment 2, PROBLEM 3](#assignment-2-problem-3), [Assignment 3, PROBLEM 1](#assignment-3-problem-1), [Assignment 3, PROBLEM 2](#assignment-3-problem-2), [Assignment 3, PROBLEM 3](#assignment-3-problem-3), [Assignment 4, PROBLEM 1](#assignment-4-problem-1), [Exam2024.PROBLEM1 – Random variable generation: rejection & inversion sampling](#exam2024-problem1-random-variable-generation-rejection-inversion-sampling), [Exam2024.PROBLEM2 – Logistic regression: calibration & predictive intervals (spam example)](#exam2024-problem2-logistic-regression-calibration-predictive-intervals-spam-example), [Exam2024.PROBLEM3 – Markov chains: classification of properties & stationary behavior](#exam2024-problem3-markov-chains-classification-of-properties-stationary-behavior), [ExamJanuary_2023.PROBLEM1 – Markov chains](#examjanuary-2023-problem1-markov-chains), [ExamJanuary_2023.PROBLEM2 – Linear regression](#examjanuary-2023-problem2-linear-regression), [ExamJanuary_2023.PROBLEM3 – Count regression (visits)](#examjanuary-2023-problem3-count-regression-visits), [Students passing exam (Sample exam problem)](#students-passing-exam-sample-exam-problem)
Print objects to stdout.
```python
print("x =", 3, "y =", 4, sep=" | ")
```

<a id="pyref-len"></a>
### `len(obj)`
**Used in:** [Assignment 1, PROBLEM 3](#assignment-1-problem-3), [Assignment 2, PROBLEM 3](#assignment-2-problem-3), [Assignment 3, PROBLEM 3](#assignment-3-problem-3), [Assignment 4, PROBLEM 1](#assignment-4-problem-1), [Exam2024.PROBLEM1 – Random variable generation: rejection & inversion sampling](#exam2024-problem1-random-variable-generation-rejection-inversion-sampling), [Exam2024.PROBLEM2 – Logistic regression: calibration & predictive intervals (spam example)](#exam2024-problem2-logistic-regression-calibration-predictive-intervals-spam-example)
Number of items in a container.
```python
len([1,2,3])
```

<a id="pyref-range"></a>
### `range(start, stop, step)`
**Used in:** [Assignment 2, PROBLEM 1](#assignment-2-problem-1), [Assignment 4, PROBLEM 1](#assignment-4-problem-1), [Students passing exam (Sample exam problem)](#students-passing-exam-sample-exam-problem)
Generate integer sequences (commonly for loops).
```python
for i in range(5):
    pass
```

<a id="pyref-enumerate"></a>
### `enumerate(iterable, start=0)`
**Used in:** [Assignment 3, PROBLEM 3](#assignment-3-problem-3), [Exam2024.PROBLEM2 – Logistic regression: calibration & predictive intervals (spam example)](#exam2024-problem2-logistic-regression-calibration-predictive-intervals-spam-example), [Students passing exam (Sample exam problem)](#students-passing-exam-sample-exam-problem)
Loop with an automatic index.
```python
for i, v in enumerate(["a","b"]):
    print(i, v)
```

<a id="pyref-isinstance"></a>
### `isinstance(obj, cls)`
**Used in:** [Assignment 1, PROBLEM 1](#assignment-1-problem-1), [Assignment 4, PROBLEM 1](#assignment-4-problem-1), [Exam2024.PROBLEM1 – Random variable generation: rejection & inversion sampling](#exam2024-problem1-random-variable-generation-rejection-inversion-sampling)
Type checking.
```python
isinstance(3.0, float)
```

<a id="pyref-open"></a>
### `open(path, mode='r', encoding=None)`
**Used in:** [Assignment 1, PROBLEM 2](#assignment-1-problem-2), [Assignment 1, PROBLEM 3](#assignment-1-problem-3), [Assignment 3, PROBLEM 1](#assignment-3-problem-1), [Assignment 4, PROBLEM 1](#assignment-4-problem-1)
Open a file handle (use `with` to auto-close).
```python
with open("file.txt", "r", encoding="utf-8") as f:
    txt = f.read()
```

<a id="pyref-sum"></a>
### `sum(iterable)`
**Used in:** [Assignment 1, PROBLEM 3](#assignment-1-problem-3), [Assignment 2, PROBLEM 1](#assignment-2-problem-1), [Assignment 2, PROBLEM 2](#assignment-2-problem-2), [Assignment 2, PROBLEM 3](#assignment-2-problem-3), [Assignment 4, PROBLEM 1](#assignment-4-problem-1), [Exam2024.PROBLEM1 – Random variable generation: rejection & inversion sampling](#exam2024-problem1-random-variable-generation-rejection-inversion-sampling), [Exam2024.PROBLEM3 – Markov chains: classification of properties & stationary behavior](#exam2024-problem3-markov-chains-classification-of-properties-stationary-behavior)
Sum numeric values.
```python
sum([1,2,3])
```

---
## 2) math module

<a id="pyref-math-log"></a>
### `math.log(x)`
**Used in:** [Assignment 4, PROBLEM 1](#assignment-4-problem-1)
Natural logarithm.
```python
import math
math.log(10)
```

<a id="pyref-math-comb"></a>
### `math.comb(n, k)`
Binomial coefficient “n choose k”.
```python
import math
math.comb(5, 2)
```

---
## 3) pandas

<a id="pyref-pd-read-csv"></a>
### `pd.read_csv(path)`
**Used in:** [Assignment 1, PROBLEM 2](#assignment-1-problem-2), [Assignment 3, PROBLEM 2](#assignment-3-problem-2), [Assignment 3, PROBLEM 3](#assignment-3-problem-3), [Exam2024.PROBLEM2 – Logistic regression: calibration & predictive intervals (spam example)](#exam2024-problem2-logistic-regression-calibration-predictive-intervals-spam-example), [ExamJanuary_2023.PROBLEM2 – Linear regression](#examjanuary-2023-problem2-linear-regression), [ExamJanuary_2023.PROBLEM3 – Count regression (visits)](#examjanuary-2023-problem3-count-regression-visits)
Read a CSV (or delimited) file into a DataFrame.
```python
import pandas as pd
df = pd.read_csv("data.csv")
```

<a id="pyref-df-head"></a>
### `df.head(n=5)`
**Used in:** [Assignment 1, PROBLEM 2](#assignment-1-problem-2), [Assignment 3, PROBLEM 2](#assignment-3-problem-2), [Assignment 3, PROBLEM 3](#assignment-3-problem-3), [Exam2024.PROBLEM2 – Logistic regression: calibration & predictive intervals (spam example)](#exam2024-problem2-logistic-regression-calibration-predictive-intervals-spam-example), [ExamJanuary_2023.PROBLEM2 – Linear regression](#examjanuary-2023-problem2-linear-regression)
Preview the first rows of a DataFrame.
```python
df.head()
```

---
## 4) csv module

<a id="pyref-csv-reader"></a>
### `csv.reader(file_obj)`
**Used in:** [Assignment 1, PROBLEM 3](#assignment-1-problem-3)
Iterate over rows in a CSV using the standard library.
```python
import csv
with open("data.csv", newline="") as f:
    for row in csv.reader(f):
        print(row)
```

---
## 5) scipy.optimize

<a id="pyref-optimize-minimize"></a>
### `optimize.minimize(fun, x0, method=...)`
**Used in:** [Assignment 2, PROBLEM 2](#assignment-2-problem-2), [Assignment 3, PROBLEM 3](#assignment-3-problem-3), [Exam2024.PROBLEM2 – Logistic regression: calibration & predictive intervals (spam example)](#exam2024-problem2-logistic-regression-calibration-predictive-intervals-spam-example)
Numerical minimization (e.g., negative log-likelihood).
```python
from scipy import optimize
res = optimize.minimize(fun, x0, method="BFGS")
```

---
## 6) scikit-learn

<a id="pyref-svc"></a>
### `SVC(kernel='linear')`
**Used in:** [Assignment 3, PROBLEM 2](#assignment-3-problem-2)
Linear Support Vector Classifier.
```python
from sklearn.svm import SVC
clf = SVC(kernel="linear")
```

<a id="pyref-pipeline"></a>
### `Pipeline(steps=[...])`
**Used in:** [Assignment 4, PROBLEM 1](#assignment-4-problem-1)
Chain preprocessing + model into one estimator.
```python
from sklearn.pipeline import Pipeline
pipe = Pipeline(steps=[("model", clf)])
```

<a id="pyref-decisiontreeregressor"></a>
### `DecisionTreeRegressor(max_depth=..., random_state=...)`
**Used in:** [Assignment 3, PROBLEM 3](#assignment-3-problem-3), [Exam2024.PROBLEM2 – Logistic regression: calibration & predictive intervals (spam example)](#exam2024-problem2-logistic-regression-calibration-predictive-intervals-spam-example)
Regression tree model.
```python
from sklearn.tree import DecisionTreeRegressor
m = DecisionTreeRegressor(max_depth=3, random_state=0)
```

---

---
## 2) NumPy / pandas / text utilities (added from ExamJanuary_2022)

<a id="pyref-np-random-default-rng-seed-none"></a>
### `np.random.default_rng(seed=None)`
**Used in:** [ExamJanuary_2022.PROBLEM2](#examjanuary-2022-problem2-random-variable-generation-and-transformation)

**What it does:** Create a NumPy random number generator (recommended over legacy `np.random`).

<a id="pyref-np-mean-a"></a>
### `np.mean(a)`
**Used in:** [ExamJanuary_2022.PROBLEM3](#examjanuary-2022-problem3-concentration-of-measure), [ExamJanuary_2022.PROBLEM4](#examjanuary-2022-problem4-sms-spam-filtering), [ExamJanuary_2022.PROBLEM6](#examjanuary-2022-problem6-black-box-testing)

**What it does:** Compute the arithmetic mean of array-like `a`.

<a id="pyref-np-sqrt-x"></a>
### `np.sqrt(x)`
**Used in:** [ExamJanuary_2022.PROBLEM3](#examjanuary-2022-problem3-concentration-of-measure)

**What it does:** Compute elementwise square root.

<a id="pyref-np-log-x"></a>
### `np.log(x)`
**Used in:** [ExamJanuary_2022.PROBLEM3](#examjanuary-2022-problem3-concentration-of-measure)

**What it does:** Compute natural logarithm elementwise.

<a id="pyref-np-array-obj"></a>
### `np.array(obj)`
**Used in:** [ExamJanuary_2022.PROBLEM5](#examjanuary-2022-problem5-markovian-travel)

**What it does:** Create a NumPy array from a Python object (list, list of lists, etc.).

<a id="pyref-np-zeros-shape"></a>
### `np.zeros(shape)`
**Used in:** [ExamJanuary_2022.PROBLEM5](#examjanuary-2022-problem5-markovian-travel)

**What it does:** Create an array filled with zeros with given shape.

<a id="pyref-np-ones-shape"></a>
### `np.ones(shape)`
**Used in:** [ExamJanuary_2022.PROBLEM5](#examjanuary-2022-problem5-markovian-travel)

**What it does:** Create an array filled with ones with given shape.

<a id="pyref-np-eye-n"></a>
### `np.eye(n)`
**Used in:** [ExamJanuary_2022.PROBLEM5](#examjanuary-2022-problem5-markovian-travel)

**What it does:** Create an `n×n` identity matrix.

<a id="pyref-np-vstack-tup"></a>
### `np.vstack(tup)`
**Used in:** [ExamJanuary_2022.PROBLEM5](#examjanuary-2022-problem5-markovian-travel)

**What it does:** Stack arrays vertically (row-wise).

<a id="pyref-np-divide-a-b-where"></a>
### `np.divide(a, b, where=...)`
**Used in:** [ExamJanuary_2022.PROBLEM5](#examjanuary-2022-problem5-markovian-travel)

**What it does:** Elementwise division with optional `where` mask to avoid division by zero.

<a id="pyref-np-linalg-lstsq-a-b-rcond-none"></a>
### `np.linalg.lstsq(A, b, rcond=None)`
**Used in:** [ExamJanuary_2022.PROBLEM5](#examjanuary-2022-problem5-markovian-travel)

**What it does:** Solve least-squares problem `Ax≈b` (useful for stationary distribution constraints).

<a id="pyref-np-linalg-matrix-power-a-k"></a>
### `np.linalg.matrix_power(A, k)`
**Used in:** [ExamJanuary_2022.PROBLEM5](#examjanuary-2022-problem5-markovian-travel)

**What it does:** Compute integer matrix power `A^k`.

<a id="pyref-pd-concat-objs-axis-0"></a>
### `pd.concat(objs, axis=0)`
**Used in:** [ExamJanuary_2022.PROBLEM5](#examjanuary-2022-problem5-markovian-travel)

**What it does:** Concatenate pandas Series/DataFrames along an axis (e.g., combine `from` and `to`).

<a id="pyref-pd-unique-values"></a>
### `pd.unique(values)`
**Used in:** [ExamJanuary_2022.PROBLEM5](#examjanuary-2022-problem5-markovian-travel)

**What it does:** Return unique values preserving order (works on Series/array-like).

<a id="pyref-text-lower"></a>
### `text.lower()`
**Used in:** [ExamJanuary_2022.PROBLEM4](#examjanuary-2022-problem4-sms-spam-filtering), [ExamJanuary_2022.PROBLEM6](#examjanuary-2022-problem6-black-box-testing)

**What it does:** Return lowercase version of a string (case-insensitive matching).

<a id="pyref-text-count-sub"></a>
### `text.count(sub)`
**Used in:** [ExamJanuary_2022.PROBLEM4](#examjanuary-2022-problem4-sms-spam-filtering)

**What it does:** Count non-overlapping occurrences of substring `sub` in a string.

<a id="pyref-zip-iterables"></a>
### `zip(*iterables)`
**Used in:** [ExamJanuary_2022.PROBLEM5](#examjanuary-2022-problem5-markovian-travel)

**What it does:** Pair elements from iterables into tuples (e.g., build transitions `(from,to)`).

```python
import numpy as np
import pandas as pd
import math
import csv
from scipy import optimize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import precision_score, recall_score

print('Reference imports loaded.')
```
