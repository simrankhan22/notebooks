# 1MS041 Data Science --- Markov Chain Questions & Answers

> **Source-based study file.** This document collects the Markov-chain
> questions, tasks, answers, explanations, formulas, and code patterns
> found in the uploaded course preparation/exam material. It focuses
> only on Markov chains and does not silently add unrelated material.

## Contents

1.  Markov-chain fundamentals
2.  Day 6 review questions and mini exam
3.  Markov-chain simulation
4.  Transition matrices and multi-step probabilities
5.  Stationary distributions
6.  Hitting and return times
7.  Irreducibility, reducibility, periodicity and aperiodicity
8.  Reversibility
9.  January 2022 --- Markovian Travel
10. January 2023 --- Markov chains
11. 2024 --- Classification of Markov chains
12. 2026 Exam vB --- Courier-company Markov chain
13. Exam-style coding templates
14. Quick revision questions
15. Markov-chain memory sheet

------------------------------------------------------------------------

# 1. Markov-chain fundamentals

## Q1. What is a Markov chain?

**Answer:** A Markov chain models a process that moves between states
over discrete time steps. The next state is determined probabilistically
from the current state through a transition matrix.

------------------------------------------------------------------------

## Q2. What is a transition matrix?

**Answer:** A transition matrix is a matrix in which entry $P_{ij}$
gives the probability of moving from state $i$ to state $j$ in one step.

------------------------------------------------------------------------

## Q3. Why must every row of a transition matrix sum to 1?

**Answer:** Each row represents all possible next states from the
current state. One of those outcomes must occur, so their probabilities
must sum to 1.

For a valid transition matrix:

$$
\sum_j P_{ij}=1
$$

for every state $i$.

In NumPy:

``` python
P.sum(axis=1)
```

should give ones.

------------------------------------------------------------------------

## Q4. What is a stationary distribution?

**Answer:** A stationary distribution is a probability distribution that
remains unchanged after a transition.

It satisfies:

$$
\pi P = \pi.
$$

It must also satisfy:

$$
\sum_i \pi_i=1
$$

and the entries must represent probabilities.

------------------------------------------------------------------------

## Q5. What does long-run behavior mean?

**Answer:** It describes the probabilities of being in each state after
a very large number of transitions. In the Day 6 material, these
probabilities are described as converging to the stationary
distribution.

------------------------------------------------------------------------

## Q6. What is hitting time?

**Answer:** Hitting time is the number of steps needed to reach a target
state for the first time.

If the target is state $j$, the hitting time is the first time the chain
reaches $j$.

------------------------------------------------------------------------

## Q7. Why can simulated state frequencies differ slightly from the theoretical distribution?

**Answer:** Simulation uses a finite number of samples, so random
variation (sampling error) occurs. As the number of simulations
increases, the simulated distribution approaches the theoretical
distribution.

------------------------------------------------------------------------

# 2. Day 6 review questions and mini exam

The uploaded Day 6 review contains the following conceptual questions.
fileciteturn6file0L80-L98

## Q8. What is a transition matrix?

**Answer:** A transition matrix is a matrix where entry $P_{ij}$ gives
the probabilities of moving from state $i$ to state $j$ in a Markov
chain.

------------------------------------------------------------------------

## Q9. Why must each row sum to 1?

**Answer:** Each row represents all possible next states from the
current state. Since one of these outcomes must occur, the probabilities
must sum to 1.

------------------------------------------------------------------------

## Q10. What is a stationary distribution?

**Answer:** It is a probability distribution that remains unchanged
after a transition and satisfies:

$$
\pi P=\pi.
$$

------------------------------------------------------------------------

## Q11. What does long-run behavior mean?

**Answer:** It describes the probabilities of being in each state after
a very large number of transitions. In the supplied Day 6 explanation,
these probabilities converge to the stationary distribution.

------------------------------------------------------------------------

## Q12. What is hitting time?

**Answer:** It is the number of steps needed to reach a target state for
the first time.

------------------------------------------------------------------------

## Q13. Why can simulation and theoretical distribution be slightly different?

**Answer:** Simulation uses a finite number of samples, causing random
variation. Increasing the number of simulations makes the empirical
distribution approach the theoretical distribution.

------------------------------------------------------------------------

## Q14. What was the Day 6 mini-exam task?

**Answer:** Create your own 2-state Markov chain and:

1.  Check row sums.
2.  Simulate 10,000 steps.
3.  Estimate empirical state frequencies.
4.  Approximate the stationary distribution.
5.  Compare the simulation and stationary distribution.

The supplied material gives exactly these five tasks.
fileciteturn6file0L91-L98

------------------------------------------------------------------------

# 3. Markov-chain simulation

## Q15. How do you simulate a Markov chain in NumPy?

**Answer:**

``` python
chain = np.zeros(T, dtype=int)
chain[0] = 0

for t in range(1, T):
    current_state = chain[t-1]
    chain[t] = rng.choice(
        [0, 1],
        p=P[current_state]
    )
```

The important idea is that the probability vector comes from the row
corresponding to the current state.

------------------------------------------------------------------------

## Q16. How do you estimate empirical state frequencies?

**Answer:**

``` python
empirical = (
    np.bincount(chain, minlength=2)
    / len(chain)
)
```

For a 3-state chain:

``` python
empirical = (
    np.bincount(chain, minlength=3)
    / len(chain)
)
```

------------------------------------------------------------------------

## Q17. What example transition matrix was used in the Day 6 mini-exam?

**Answer:**

$$
P=
\begin{pmatrix}
0.8 & 0.2\\
0.4 & 0.6
\end{pmatrix}.
$$

The supplied task was to simulate 10,000 steps from state 0, estimate
empirical frequencies, approximate the stationary distribution, and
compare them. fileciteturn4file3L312-L355

------------------------------------------------------------------------

## Q18. How do you update a probability distribution one step at a time?

**Answer:**

``` python
pi = pi @ P
```

This is the distribution-update rule shown in the Day 6 material.
fileciteturn4file3L324-L334

------------------------------------------------------------------------

# 4. Transition matrices and multi-step probabilities

## Q19. How do you calculate the probability of moving between states after multiple steps?

**Answer:** Use powers of the transition matrix.

For two steps:

$$
P^2.
$$

For three steps:

$$
P^3.
$$

In NumPy:

``` python
P2 = np.linalg.matrix_power(P, 2)
P3 = np.linalg.matrix_power(P, 3)
```

Then `P2[i, j]` is the probability of being in state $j$ after two steps
when starting in state $i$.

------------------------------------------------------------------------

## Q20. What is the difference between "in state after two steps" and "first time in state after two steps"?

**Answer:** They are different questions.

### After two steps

You count **all paths** that end in the target state at step 2, even if
the chain visited that target earlier.

### First time after two steps

You exclude paths that reached the target before step 2.

This distinction is explicitly tested in the January 2023 and earlier
Markov-chain problems.

------------------------------------------------------------------------

# 5. Stationary distributions

## Q21. How do you calculate a stationary distribution mathematically?

**Answer:** Solve:

$$
\pi P=\pi
$$

together with:

$$
\sum_i\pi_i=1.
$$

Equivalently:

$$
(P^T-I)\pi=0.
$$

The supplied solution uses the normalization constraint to solve the
linear system. fileciteturn5file1L277-L317

------------------------------------------------------------------------

## Q22. How do you calculate a stationary distribution using `np.linalg.lstsq`?

**Answer:**

``` python
n = P.shape[0]

A = P.T - np.eye(n)
A = np.vstack([A, np.ones(n)])

b = np.zeros(n + 1)
b[-1] = 1

pi = np.linalg.lstsq(
    A,
    b,
    rcond=None
)[0]
```

Then check:

``` python
pi.sum()
```

which should be 1.

------------------------------------------------------------------------

## Q23. How can you find a stationary distribution using eigenvectors?

**Answer:** Find the eigenvector of $P^T$ corresponding to eigenvalue 1,
then normalize it so its entries sum to 1.

The supplied code is:

``` python
eigenvalues, eigenvectors = np.linalg.eig(P.T)

idx = np.argmin(
    np.abs(eigenvalues - 1)
)

stationary = eigenvectors[:, idx].real
stationary = stationary / stationary.sum()
```

This method appears in the uploaded Markov-chain solutions.
fileciteturn8file2L392-L410

------------------------------------------------------------------------

## Q24. What should you check after calculating a stationary distribution?

**Answer:**

1.  Its entries should represent probabilities.
2.  The entries should sum to 1.
3.  It should satisfy $\pi P=\pi$.

------------------------------------------------------------------------

# 6. Hitting and return times

## Q25. What is a hitting time?

**Answer:** The hitting time is the first time a chain reaches a target
state.

The Day 6 example defines it as "how long until the chain first reaches
a target state." fileciteturn6file0L26-L37

------------------------------------------------------------------------

## Q26. How do you simulate a hitting time?

**Answer:**

``` python
current_state = 0
t = 0

while current_state != target_state:
    current_state = rng.choice(
        states,
        p=P[current_state]
    )
    t += 1
```

The supplied Day 6 implementation also uses a safety limit:

``` python
max_steps = 1000
```

and records the hitting times over many simulated chains.
fileciteturn6file0L44-L66

------------------------------------------------------------------------

## Q27. What hitting-time simulation example was given?

**Answer:** Target state:

``` text
Rainy = 2
```

Starting state:

``` text
Sunny = 0
```

Number of simulated chains:

``` text
R = 5000
```

The supplied simulation estimated the mean hitting time as:

``` text
6.692
```

and plotted a histogram of the hitting times.
fileciteturn6file0L26-L77

------------------------------------------------------------------------

## Q28. What is the first-step equation for expected hitting time?

**Answer:** If $h(i)$ is the expected time to reach a target, then for a
non-target state:

$$
h(i)
=
1+\sum_j P_{ij}h(j).
$$

For the target state:

$$
h(\text{target})=0.
$$

The uploaded January 2023 solution explicitly uses this first-step
equation. fileciteturn4file9L715-L724

------------------------------------------------------------------------

## Q29. How can the expected hitting time be calculated using a linear system?

**Answer:** If the target state is removed from the transient-state
system, write:

$$
(I-Q)h=\mathbf{1}.
$$

Then solve:

``` python
h = np.linalg.solve(
    I - Q,
    ones
)
```

This exact approach appears in the supplied Markov-chain solution.
fileciteturn5file1L333-L345

------------------------------------------------------------------------

## Q30. What is the expected return time to a state when the stationary distribution is known?

**Answer:** The supplied solution uses:

$$
E[\text{return time to state }i]
=
\frac{1}{\pi_i}.
$$

The January 2023 solution explicitly states this relationship.
fileciteturn8file2L414-L423

------------------------------------------------------------------------

# 7. Irreducibility, reducibility, periodicity and aperiodicity

## Q31. What does it mean for two states to communicate?

**Answer:** Two states $i$ and $j$ communicate if each is reachable from
the other.

The uploaded solution states this definition when explaining
irreducibility. fileciteturn4file9L876-L881

------------------------------------------------------------------------

## Q32. What is an irreducible Markov chain?

**Answer:** A Markov chain is irreducible when all states communicate
with each other.

In a finite chain, this means every state can eventually be reached from
every other state.

------------------------------------------------------------------------

## Q33. How can you quickly recognize an irreducible chain from a transition matrix?

**Answer:** In one of the supplied examples, every entry of the
transition matrix is positive. Therefore, every state can reach every
other state in one step, so all states communicate and the chain is
irreducible. fileciteturn8file2L359-L366

------------------------------------------------------------------------

## Q34. What is a reducible Markov chain?

**Answer:** A chain is reducible if not every state communicates with
every other state.

The 2024 problem explicitly asks whether each supplied chain is
irreducible/reducible. fileciteturn8file1L212-L220

------------------------------------------------------------------------

## Q35. What is the period of a state?

**Answer:** The period describes the possible numbers of steps in which
a state can return to itself. In the 2024 material, state periods are
explicitly requested for each chain.

------------------------------------------------------------------------

## Q36. What does aperiodic mean in the supplied exam material?

**Answer:** A chain is aperiodic when the states have period 1. In the
2024 answer set, chains marked aperiodic have all their state periods
equal to 1. fileciteturn8file1L223-L238

------------------------------------------------------------------------

## Q37. What is the relationship between a self-loop and aperiodicity?

**Answer:** The supplied 2024 matrices include states with positive
self-transition probabilities. Such a self-loop gives a return in one
step, making the state's period 1. This is reflected in the supplied
period answers.

------------------------------------------------------------------------

# 8. Reversibility

## Q38. What does it mean for a Markov chain to be reversible?

**Answer:** The supplied solution tests the detailed-balance condition:

$$
\pi_iP_{ij}
=
\pi_jP_{ji}.
$$

for all pairs of states.

------------------------------------------------------------------------

## Q39. How is reversibility checked in the supplied code?

**Answer:**

``` python
lhs = pi.reshape(-1, 1) * P
rhs = pi.reshape(1, -1) * P.T

np.allclose(
    lhs,
    rhs,
    atol=1e-8,
    rtol=0
)
```

This is exactly the detailed-balance test used in the 2024 material.
fileciteturn8file1L281-L298

------------------------------------------------------------------------

# 9. January 2022 --- Markovian Travel

## Q40. What was the January 2022 Markovian Travel problem about?

**Answer:** It used the simulated **Travel Dataset --- Datathon 2019**,
stored as:

``` text
data/flights.csv
```

The dataset represents corporate travel involving flights and hotels.
Each row represents a flight from one city to another.
fileciteturn8file0L12-L20

------------------------------------------------------------------------

## Q41. What Markov-chain model was assumed for the travel data?

**Answer:** Each user's travel trajectory was modeled as a realization
of a stationary Markov chain:

$$
X_t=\text{current city at step }t
$$

and

$$
X_{t+1}=\text{city travelled to at the next step}.
$$

The transition probability was assumed to satisfy:

$$
P(X_{t+1}=y\mid X_t=x)=p(x,y)
$$

for all $x,y$. fileciteturn8file0L17-L22

------------------------------------------------------------------------

## Q42. What were the four tasks in the January 2022 Markovian Travel problem?

**Answer:**

1.  Load `data/flights.csv` and fill in the requested variables.
2.  Estimate the Markov-chain transition matrix for the users' travels.
3.  Use the transition matrix to calculate the stationary distribution.
4.  Starting in **Aracaju (SE)**, calculate the probability of being
    back in **Aracaju (SE)** after 3 steps.

These are the four tasks in the supplied exam.
fileciteturn8file0L17-L23

------------------------------------------------------------------------

## Q43. How were the cities indexed?

**Answer:** The cities were sorted alphabetically. The first city,
`'Aracaju (SE)'`, corresponds to index 0.

The code constructs:

``` python
unique_cities = sorted(set(cities))

indexToCity = {
    i: city
    for i, city in enumerate(unique_cities)
}

cityToIndex = {
    city: i
    for i, city in enumerate(unique_cities)
}
```

fileciteturn8file0L63-L82

------------------------------------------------------------------------

## Q44. How were transition counts constructed?

**Answer:**

``` python
transitions = list(
    zip(df["from"], df["to"])
)

transition_counts = makeFreqDict(
    transitions
)
```

Each `(from_city, to_city)` pair is counted.

------------------------------------------------------------------------

## Q45. How was the transition matrix estimated?

**Answer:** Start with a zero matrix:

``` python
transition_matrix = np.zeros(
    (n_cities, n_cities)
)
```

Then add the transition counts:

``` python
for (city_from, city_to), count in transition_counts.items():
    i = cityToIndex[city_from]
    j = cityToIndex[city_to]
    transition_matrix[i, j] += count
```

Finally normalize each row:

``` python
row_sums = transition_matrix.sum(
    axis=1,
    keepdims=True
)

transition_matrix = np.divide(
    transition_matrix,
    row_sums,
    where=row_sums != 0
)
```

This is the supplied maximum-likelihood transition-matrix construction.
fileciteturn8file0L84-L108

------------------------------------------------------------------------

## Q46. How was the stationary distribution calculated for the travel chain?

**Answer:** Solve:

$$
\pi P=\pi
$$

with:

$$
\sum_i\pi_i=1.
$$

The supplied code forms:

``` python
A = transition_matrix.T - np.eye(n)
A = np.vstack([A, np.ones(n)])

b = np.zeros(n + 1)
b[-1] = 1

stationary_distribution_problem5 = (
    np.linalg.lstsq(
        A,
        b,
        rcond=None
    )[0]
)
```

fileciteturn8file0L111-L129

------------------------------------------------------------------------

## Q47. How was the 3-step return probability to Aracaju calculated?

**Answer:**

``` python
i = cityToIndex['Aracaju (SE)']

transition_matrix_3 = np.linalg.matrix_power(
    transition_matrix,
    3
)

return_probability = (
    transition_matrix_3[i, i]
)
```

So the required probability is the diagonal entry corresponding to
Aracaju in $P^3$. fileciteturn8file0L132-L140

------------------------------------------------------------------------

# 10. January 2023 --- Markov chains

## Q48. What was the January 2023 Markov-chain scenario?

**Answer:** A courier company has trucks moving between three regions:

-   Downtown
-   Suburbs
-   Countryside

The transition probabilities are:

  Current region     Downtown   Suburbs   Countryside
  ---------------- ---------- --------- -------------
  Downtown                0.3       0.4           0.3
  Suburbs                 0.2       0.5           0.3
  Countryside             0.4       0.3           0.3

fileciteturn6file6L452-L468

------------------------------------------------------------------------

## Q49. January 2023 --- Part 1: What is the probability of being in Downtown after two steps starting from Suburbs?

**Answer:** Calculate:

$$
P^2
$$

and take:

``` python
P2[1, 0]
```

because Suburbs is row 1 and Downtown is column 0.

The supplied answer is:

``` text
0.28
```

fileciteturn7file1L85-L104

------------------------------------------------------------------------

## Q50. January 2023 --- Part 2: What is the probability of being in Downtown for the first time after two steps starting from Suburbs?

**Answer:** Only paths that first reach Downtown at step 2 count.

The supplied solution considers:

``` text
Suburbs → Countryside → Downtown
```

and

``` text
Suburbs → Suburbs → Downtown
```

and excludes:

``` text
Suburbs → Downtown → Downtown
```

because that path reaches Downtown at step 1.

The supplied answer is:

``` text
0.22
```

The code is:

``` python
path1 = (
    P[suburbs, countryside]
    * P[countryside, downtown]
)

path2 = (
    P[suburbs, suburbs]
    * P[suburbs, downtown]
)

total_probability = path1 + path2
```

fileciteturn8file2L340-L355

------------------------------------------------------------------------

## Q51. January 2023 --- Part 3: Is the chain irreducible?

**Answer:** Yes.

The supplied explanation is that every entry in the transition matrix is
positive, so every state can reach every other state in one step.
Therefore, all states communicate and the chain is irreducible.

``` python
problem1_irreducible = True
```

fileciteturn8file2L359-L366

------------------------------------------------------------------------

## Q52. January 2023 --- Part 4: What is the stationary distribution?

**Answer:** Solve:

$$
\pi P=\pi
$$

with the normalization:

$$
\sum_i\pi_i=1.
$$

The supplied solution gives two methods: - solve the constrained linear
system, - find the eigenvector of $P^T$ corresponding to eigenvalue 1
and normalize it.

fileciteturn8file2L370-L410

------------------------------------------------------------------------

## Q53. January 2023 --- Part 5: What is the expected number of steps until first entering Suburbs when starting from Downtown?

**Answer:** Let $h(i)$ be the expected hitting time of Suburbs.

Set:

$$
h(\text{Suburbs})=0.
$$

For every other state:

$$
h(i)=1+\sum_jP_{ij}h(j).
$$

The supplied answer recommends solving the resulting linear system, and
the exam says that probabilities for hitting times below 30 are
sufficient to get within 1 decimal point. Simulation could also be used,
but receives a lower maximum score. fileciteturn6file7L559-L563

The supplied solution also gives the matrix form:

``` python
A = np.array([
    [1 - P[0,0], -P[0,2]],
    [-P[2,0],    1 - P[2,2]]
])

b = np.array([1.0, 1.0])

hD, hC = np.linalg.solve(A, b)
```

The expected hitting time from Downtown is `hD`.
fileciteturn4file7L692-L705

------------------------------------------------------------------------

# 11. 2024 --- Classification of Markov chains

## Q54. What did the 2024 Markov-chain exam problem ask?

**Answer:** Four Markov chains, A, B, C and D, were supplied as
diagrams. For **each chain**, students had to answer:

1.  What is the transition matrix?
2.  Is the chain irreducible?
3.  Is it aperiodic, and what is the period of each state?
4.  Does it have a stationary distribution, and if so, what is it?
5.  Is it reversible?

This was a 13-point Markov-chain problem.
fileciteturn4file6L579-L595

------------------------------------------------------------------------

## Q55. What transition matrix was supplied/constructed for chain A?

**Answer:**

``` python
problem3_A = np.array([
    [0.8, 0.2, 0,   0],
    [0.6, 0.2, 0.2, 0],
    [0,   0.4, 0,   0.6],
    [0,   0,   0.8, 0.2]
])
```

fileciteturn8file1L178-L189

------------------------------------------------------------------------

## Q56. What transition matrix was supplied/constructed for chain B?

**Answer:**

``` python
problem3_B = np.array([
    [0,   0.2, 0,   0.8],
    [0,   0,   1,   0],
    [0,   1,   0,   0],
    [0.5, 0,   0.5, 0]
])
```

fileciteturn8file1L190-L195

------------------------------------------------------------------------

## Q57. What transition matrix was supplied/constructed for chain C?

**Answer:**

``` python
problem3_C = np.array([
    [0.2, 0.3, 0,   0,   0.5],
    [0.2, 0.2, 0.6, 0,   0],
    [0,   0.4, 0,   0.6, 0],
    [0,   0,   0,   0.6, 0.4],
    [0,   0,   0,   0.4, 0.6]
])
```

fileciteturn8file1L196-L202

------------------------------------------------------------------------

## Q58. What transition matrix was supplied/constructed for chain D?

**Answer:**

``` python
problem3_D = np.array([
    [0.8, 0.2, 0,   0],
    [0.6, 0.2, 0.2, 0],
    [0,   0.4, 0,   0.6],
    [0.1, 0,   0.7, 0.2]
])
```

fileciteturn8file1L203-L208

------------------------------------------------------------------------

## Q59. Which of the 2024 chains were irreducible?

**Answer:**

``` text
A: True
B: False
C: False
D: True
```

These are the supplied autograder answers.
fileciteturn7file2L145-L152

------------------------------------------------------------------------

## Q60. Which of the 2024 chains were aperiodic?

**Answer:**

``` text
A: True
B: False
C: True
D: True
```

The supplied state periods were:

``` text
A: [1, 1, 1, 1]
B: [2, 2, 2, 2]
C: [1, 1, 1, 1, 1]
D: [1, 1, 1, 1]
```

fileciteturn7file2L155-L170

------------------------------------------------------------------------

## Q61. Did the 2024 chains have stationary distributions?

**Answer:** The supplied answer marks **all four chains as having
stationary distributions**:

``` text
A: True
B: True
C: True
D: True
```

fileciteturn8file1L242-L249

------------------------------------------------------------------------

## Q62. How were the 2024 stationary distributions calculated?

**Answer:**

``` python
def stationary_distribution(P):
    P = np.asarray(P, dtype=float)
    n = P.shape[0]

    A = P.T - np.eye(n)
    b = np.zeros(n)

    A[-1, :] = 1.0
    b[-1] = 1.0

    pi = np.linalg.solve(A, b)
    pi = np.clip(pi, 0.0, 1.0)
    pi = pi / pi.sum()

    return pi
```

Then:

``` python
pi_ex_A = stationary_distribution(problem3_A)
pi_ex_B = stationary_distribution(problem3_B)
pi_ex_C = stationary_distribution(problem3_C)
pi_ex_D = stationary_distribution(problem3_D)
```

fileciteturn8file1L251-L278

------------------------------------------------------------------------

## Q63. How was reversibility checked in the 2024 problem?

**Answer:** By checking detailed balance:

$$
\pi_iP_{ij}=\pi_jP_{ji}.
$$

The supplied implementation is:

``` python
def is_reversible(P, pi=None, tol=1e-8):
    P = np.asarray(P, dtype=float)

    if pi is None:
        pi = stationary_distribution(P)

    lhs = pi.reshape(-1, 1) * P
    rhs = pi.reshape(1, -1) * P.T

    return bool(
        np.allclose(
            lhs,
            rhs,
            atol=tol,
            rtol=0
        )
    )
```

fileciteturn8file1L281-L298

> **Source limitation:** the supplied extract shows the code that
> calculates the A/B/C/D reversibility answers, but does not expose a
> final printed True/False result for each chain in the visible source
> range. Therefore this file does not invent those four final values.

------------------------------------------------------------------------

# 12. 2026 Exam vB --- Courier-company Markov chain

## Q64. What was the 2026 Exam vB Markov-chain scenario?

**Answer:** A courier company monitors trucks across four states:

-   Downtown (D)
-   Suburbs (S)
-   Countryside (C)
-   Maintenance (M)

The transition matrix is:

$$
P=
\begin{pmatrix}
0.25&0.35&0.30&0.10\\
0.20&0.40&0.30&0.10\\
0.15&0.35&0.40&0.10\\
0&0&0&1
\end{pmatrix}.
$$

fileciteturn4file2L186-L218

------------------------------------------------------------------------

## Q65. What were the six questions in the 2026 courier Markov-chain problem?

**Answer:**

1.  Starting in Suburbs, what is the probability of eventually ending up
    in Maintenance?
2.  Starting in Downtown, what is the probability of being in
    Countryside after five steps?
3.  Starting in Downtown, what is the expected number of steps before
    entering Maintenance?
4.  Is the chain irreducible? Is it aperiodic?
5.  Does the chain have a stationary distribution? If yes, compute it;
    otherwise explain why.
6.  Starting in Countryside, what is the probability that the last state
    visited before reaching Maintenance is Suburbs?

These six tasks are explicitly listed in the supplied exam.
fileciteturn4file2L206-L219

------------------------------------------------------------------------

## Q66. Why is Maintenance special in the 2026 chain?

**Answer:** Maintenance is absorbing because:

``` text
M → M = 1
```

and all other transitions out of Maintenance have probability 0.

The supplied matrix therefore has the final row:

``` python
[0.00, 0.00, 0.00, 1.00]
```

fileciteturn4file2L197-L204

------------------------------------------------------------------------

## Q67. January 2026 vB --- Part 1: What is the probability that a truck starting in Suburbs eventually reaches Maintenance?

**Answer:** The problem asks for this probability as a decimal.

Because Maintenance is absorbing, this is an absorption-probability
problem.

The supplied matrix is:

``` python
P = np.array([
    [0.25, 0.35, 0.30, 0.10],
    [0.20, 0.40, 0.30, 0.10],
    [0.15, 0.35, 0.40, 0.10],
    [0.00, 0.00, 0.00, 1.00]
])
```

fileciteturn4file2L221-L235

------------------------------------------------------------------------

## Q68. January 2026 vB --- Part 2: What is the probability of being in Countryside after five steps starting from Downtown?

**Answer:** Use:

``` python
P5 = np.linalg.matrix_power(P, 5)

probability = P5[0, 2]
```

because Downtown is state 0 and Countryside is state 2.

The supplied exam asks for exactly this five-step transition
probability. fileciteturn4file2L206-L208

------------------------------------------------------------------------

## Q69. January 2026 vB --- Part 3: How do you calculate expected time before entering Maintenance?

**Answer:** Use first-step analysis.

For each transient state:

$$
E_i
=
1+
\sum_jP_{ij}E_j.
$$

For Maintenance:

$$
E_M=0.
$$

The exam explicitly gives the first-step-analysis hint:

> Expected time from a state = 1 + sum of transition probability ×
> expected time from next state.

fileciteturn4file2L208-L214

------------------------------------------------------------------------

## Q70. January 2026 vB --- Part 4: Is the chain irreducible?

**Answer:** No.

Maintenance is absorbing and cannot transition back to Downtown,
Suburbs, or Countryside. Therefore, not all states communicate with each
other.

So the chain is **reducible**.

------------------------------------------------------------------------

## Q71. January 2026 vB --- Part 4: Is the chain aperiodic?

**Answer:** The supplied matrix contains self-loops, including:

``` text
D → D = 0.25
S → S = 0.40
C → C = 0.40
M → M = 1
```

The exam asks for the aperiodicity classification. The presence of
one-step returns is the key feature used in the course material for
period 1.

------------------------------------------------------------------------

## Q72. January 2026 vB --- Part 5: Does the chain have a stationary distribution?

**Answer:** Yes.

The supplied solution explains that the chain contains an absorbing
Maintenance state. Once the process reaches Maintenance, it cannot
leave, so probability mass can accumulate there.

The supplied stationary distribution is:

$$
\pi=[0,0,0,1].
$$

It satisfies:

$$
\pi P=\pi.
$$

fileciteturn4file8L769-L787

------------------------------------------------------------------------

## Q73. Why is `[0, 0, 0, 1]` a stationary distribution?

**Answer:** If all probability mass is placed on Maintenance, one
transition leaves it at Maintenance with probability 1. Therefore:

$$
[0,0,0,1]P
=
[0,0,0,1].
$$

This is exactly the explanation in the supplied free-text answer.
fileciteturn4file8L779-L787

------------------------------------------------------------------------

## Q74. January 2026 vB --- Part 6: What is being asked by "the last state visited before reaching Maintenance"?

**Answer:** Starting from Countryside, the question asks for the
probability that the state immediately before the final transition into
Maintenance is **Suburbs**.

It is not asking for the probability of ever visiting Suburbs. It asks
specifically which transient state the chain occupies immediately before
entering the absorbing Maintenance state.

fileciteturn4file8L790-L839

------------------------------------------------------------------------

## Q75. How is the last-state-before-absorption probability calculated?

**Answer:** The supplied solution defines a function that:

1.  identifies transient states,
2.  constructs a linear system,
3.  treats a transition into Maintenance from the target last state as
    success,
4.  solves the system.

The core setup is:

``` python
A = np.eye(len(transient_states))
b = np.zeros(len(transient_states))
```

For transitions to the absorbing state:

``` python
if i == target_last_state:
    b[row] += P[i, j]
```

For transitions among transient states:

``` python
A[row, col] -= P[i, j]
```

Then:

``` python
f = np.linalg.solve(A, b)
```

The desired answer is the entry corresponding to the starting state.
fileciteturn4file8L797-L839

------------------------------------------------------------------------

# 13. Exam-style coding templates

## Q76. What is the standard transition-matrix check?

**Answer:**

``` python
print(P.sum(axis=1))
```

Every row should sum to 1.

------------------------------------------------------------------------

## Q77. What is the standard multi-step probability calculation?

**Answer:**

``` python
Pk = np.linalg.matrix_power(P, k)

probability = Pk[start_state, target_state]
```

------------------------------------------------------------------------

## Q78. What is the standard stationary-distribution calculation using a linear system?

**Answer:**

``` python
n = P.shape[0]

A = P.T - np.eye(n)
A = np.vstack([
    A,
    np.ones(n)
])

b = np.zeros(n + 1)
b[-1] = 1

pi = np.linalg.lstsq(
    A,
    b,
    rcond=None
)[0]
```

------------------------------------------------------------------------

## Q79. What is the standard eigenvector method for stationary distribution?

**Answer:**

``` python
eigenvalues, eigenvectors = np.linalg.eig(P.T)

idx = np.argmin(
    np.abs(eigenvalues - 1)
)

pi = eigenvectors[:, idx].real
pi = pi / pi.sum()
```

------------------------------------------------------------------------

## Q80. What is the standard hitting-time simulation?

**Answer:**

``` python
hitting_times = []

for r in range(R):
    current_state = start_state
    t = 0

    while (
        current_state != target_state
        and t < max_steps
    ):
        current_state = rng.choice(
            states,
            p=P[current_state]
        )
        t += 1

    hitting_times.append(t)

mean_hitting_time = np.mean(
    hitting_times
)
```

------------------------------------------------------------------------

## Q81. What is the standard first-step-analysis equation?

**Answer:**

``` python
# target state has h[target] = 0

# for every other state i:
h[i] = 1 + sum(
    P[i, j] * h[j]
    for j in states
)
```

In matrix form for transient states:

``` python
h = np.linalg.solve(
    np.eye(Q.shape[0]) - Q,
    np.ones(Q.shape[0])
)
```

------------------------------------------------------------------------

## Q82. How do you test reversibility?

**Answer:**

``` python
lhs = pi.reshape(-1, 1) * P
rhs = pi.reshape(1, -1) * P.T

is_reversible = np.allclose(
    lhs,
    rhs,
    atol=1e-8,
    rtol=0
)
```

------------------------------------------------------------------------

# 14. Quick revision questions

## Q83. What equation defines a stationary distribution?

**Answer:**

$$
\boxed{\pi P=\pi}
$$

with:

$$
\boxed{\sum_i\pi_i=1}.
$$

------------------------------------------------------------------------

## Q84. What does $P^2$ represent?

**Answer:** Two-step transition probabilities.

`P2[i,j]` is the probability of being in state `j` after two transitions
starting from state `i`.

------------------------------------------------------------------------

## Q85. What does $P^k$ represent?

**Answer:** The transition probabilities after $k$ steps.

------------------------------------------------------------------------

## Q86. What is the difference between a stationary distribution and empirical state frequencies?

**Answer:** The stationary distribution is the theoretical distribution
satisfying $\pi P=\pi$. Empirical state frequencies are estimates
obtained from a finite simulation.

------------------------------------------------------------------------

## Q87. What is the difference between hitting time and return time?

**Answer:**

-   **Hitting time:** first time reaching a target state.
-   **Return time:** time required to return to a state that was the
    starting state.

------------------------------------------------------------------------

## Q88. What is an absorbing state?

**Answer:** A state that, once entered, cannot be left.

Its transition probability to itself is 1.

In the 2026 courier example:

``` text
Maintenance → Maintenance = 1.
```

------------------------------------------------------------------------

## Q89. Does a reducible chain necessarily have no stationary distribution?

**Answer:** No. The supplied 2026 example is reducible but **does have**
a stationary distribution:

$$
[0,0,0,1].
$$

The 2024 examples also show that the course tests stationary
distributions separately from irreducibility.

------------------------------------------------------------------------

## Q90. Does an irreducible chain automatically mean every state has period 1?

**Answer:** No. Irreducibility and aperiodicity are separate properties.
The 2024 material explicitly asks for both properties separately. Chain
B is marked irreducible = False and aperiodic = False, while other
chains have different combinations. fileciteturn7file2L145-L170

------------------------------------------------------------------------

## Q91. How can you recognize an absorbing state from a transition matrix?

**Answer:** Look for a row of the form:

``` text
[0, ..., 0, 1, 0, ..., 0]
```

where the `1` is in the state's own column.

------------------------------------------------------------------------

## Q92. What is detailed balance?

**Answer:**

$$
\boxed{\pi_iP_{ij}=\pi_jP_{ji}}
$$

for every pair of states.

It is the condition used by the supplied code to test reversibility.

------------------------------------------------------------------------

## Q93. How do you find the probability of being in a state after 3 steps?

**Answer:**

``` python
P3 = np.linalg.matrix_power(P, 3)

prob = P3[start_state, target_state]
```

This is exactly the method used for the January 2022 Aracaju return
problem. fileciteturn8file0L132-L140

------------------------------------------------------------------------

## Q94. How do you calculate a "first time at step 2" probability?

**Answer:** Enumerate only paths that: 1. do not reach the target at
step 1, 2. reach the target at step 2.

For the January 2023 example, the valid paths were:

``` text
Suburbs → Countryside → Downtown
Suburbs → Suburbs → Downtown
```

and the invalid path was:

``` text
Suburbs → Downtown → Downtown
```

because Downtown was reached at step 1. fileciteturn8file2L340-L355

------------------------------------------------------------------------

# 15. Markov-chain memory sheet

## Transition matrix

$$
P_{ij}=P(X_{t+1}=j\mid X_t=i)
$$

Every row sums to 1.

``` python
P.sum(axis=1)
```

------------------------------------------------------------------------

## Multi-step transitions

$$
P^k
$$

``` python
Pk = np.linalg.matrix_power(P, k)
```

------------------------------------------------------------------------

## Stationary distribution

$$
\boxed{\pi P=\pi}
$$

and:

$$
\boxed{\sum_i\pi_i=1}.
$$

------------------------------------------------------------------------

## Stationary distribution via eigenvector

Find eigenvector of $P^T$ for eigenvalue 1:

``` python
eigvals, eigvecs = np.linalg.eig(P.T)
idx = np.argmin(np.abs(eigvals - 1))
pi = eigvecs[:, idx].real
pi = pi / pi.sum()
```

------------------------------------------------------------------------

## Hitting time

$$
T_j=\min\{t\ge 0:X_t=j\}.
$$

------------------------------------------------------------------------

## Expected hitting time

$$
h(j)=0
$$

for target $j$, and:

$$
\boxed{
h(i)=1+\sum_kP_{ik}h(k)
}
$$

for other states.

------------------------------------------------------------------------

## Return time

For the relationship explicitly used in the supplied solution:

$$
\boxed{
E[\text{return time to state }i]
=
\frac{1}{\pi_i}
}
$$

------------------------------------------------------------------------

## Irreducible

All states communicate.

------------------------------------------------------------------------

## Reducible

Not all states communicate.

------------------------------------------------------------------------

## Period

Determined by possible return times to a state.

------------------------------------------------------------------------

## Aperiodic

Period 1.

------------------------------------------------------------------------

## Reversible

Detailed balance:

$$
\boxed{
\pi_iP_{ij}=\pi_jP_{ji}
}
$$

------------------------------------------------------------------------

## Absorbing state

A state that cannot be left once entered.

------------------------------------------------------------------------

# 16. Most important exam questions to practice

1.  **Given a transition matrix, check that it is valid.**
2.  **Simulate a Markov chain and calculate empirical state
    frequencies.**
3.  **Calculate $P^2$, $P^3$, or $P^k$.**
4.  **Find a probability of being in a state after $k$ steps.**
5.  **Distinguish "after $k$ steps" from "for the first time at step
    $k$."**
6.  **Find the stationary distribution from $\pi P=\pi$.**
7.  **Find a stationary distribution using `np.linalg.lstsq`.**
8.  **Find a stationary distribution using an eigenvector of $P^T$.**
9.  **Determine whether a chain is irreducible.**
10. **Determine whether a chain is reducible.**
11. **Determine the period of each state.**
12. **Determine whether a chain is aperiodic.**
13. **Identify absorbing states.**
14. **Calculate an expected hitting time using first-step analysis.**
15. **Calculate/remember the supplied return-time relationship
    $1/\pi_i$.**
16. **Test reversibility using detailed balance.**
17. **Estimate a transition matrix from observed state-to-state
    transitions.**
18. **Calculate a stationary distribution from an estimated transition
    matrix.**
19. **Calculate a return probability such as $P^3_{ii}$.**
20. **Handle an absorbing chain and calculate probabilities associated
    with the final transition into the absorbing state.**

------------------------------------------------------------------------

# 17. Source coverage

This Markov-chain study file is based on the uploaded course/exam
material, including:

-   **Day 6 --- Markov chain, transition matrix, simulation, state
    probabilities, stationary distribution, hitting time and long-run
    behavior**
-   **January 2022 --- Markovian Travel**
-   **January 2023 --- Markov chains**
-   **2024 --- Markov chains: classification of properties and
    stationary behavior**
-   **2026 Exam vB --- courier-company Markov chain**
-   Markov-related Python reference material used by the supplied
    solutions

The preparation file's exam index explicitly identifies January 2022
Problem 5, January 2023 Problem 1, and Exam 2024 Problem 3 as
Markov-chain problems. fileciteturn6file2L202-L214

> **Important:** Where the visible source material supplied code for an
> answer but did not expose the final numerical output, this document
> keeps the method rather than inventing a number.
