# Markov Chains — Python + Explanation

## 1. What is a Markov Chain?

A Markov chain is a stochastic process where the probability of the next state depends only on the current state, not the entire history.

A transition matrix `P` stores the probability of moving from one state to another.

Example:

    P =
    [[0.6, 0.3, 0.1],
     [0.2, 0.5, 0.3],
     [0.4, 0.1, 0.5]]

Rows = current state.
Columns = next state.

For example, `P[0,1] = 0.3` means there is a 30% probability of moving from state 0 to state 1.

Every row of a valid transition matrix must sum to 1.

---

## 2. Load the Dataset

Assume the dataset contains a column called `state`, for example:

    state
    A
    A
    B
    C
    B
    A
    C

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset.csv")

# Get observed states
states = df["state"].values

# Find unique states
unique_states = np.sort(df["state"].unique())

# Number of states
n_states = len(unique_states)

print("States:", unique_states)
print("Number of states:", n_states)
```

---

## 3. Create the Transition Count Matrix

We look at consecutive observations:

    state[t] -> state[t+1]

and count how often every transition occurs.

```python
# Empty transition count matrix
counts = np.zeros((n_states, n_states))

# Map state names to matrix indices
state_to_index = {
    state: i for i, state in enumerate(unique_states)
}

# Count transitions
for i in range(len(states) - 1):
    current = state_to_index[states[i]]
    next_state = state_to_index[states[i + 1]]

    counts[current, next_state] += 1

print("Transition count matrix:")
print(counts)
```

For example:

    [[6, 3, 1],
     [2, 5, 3],
     [4, 1, 5]]

means state A was followed by A six times, B three times, and C one time.

---

## 4. Convert Counts to a Transition Matrix

The transition matrix contains probabilities rather than counts.

For each row:

    probability = transition count / total transitions from that state

```python
P = counts / counts.sum(axis=1, keepdims=True)

print("Transition matrix:")
print(P)
```

Example:

    [[0.6, 0.3, 0.1],
     [0.2, 0.5, 0.3],
     [0.4, 0.1, 0.5]]

---

## 5. Check That Every Row Sums to 1

A valid transition matrix must have:

    sum of each row = 1

```python
row_sums = P.sum(axis=1)

print("Row sums:")
print(row_sums)

assert np.allclose(row_sums, 1)

print("Every row sums to 1!")
```

`np.allclose()` handles tiny floating-point rounding errors.

---

## 6. Simulate One Markov Chain of Length 100

We start from a state and repeatedly use the corresponding row of `P` to randomly choose the next state.

```python
length = 100

# Start in state 0
current_state = 0

# Store chain
chain = [current_state]

# Generate remaining states
for _ in range(length - 1):
    current_state = np.random.choice(
        n_states,
        p=P[current_state]
    )

    chain.append(current_state)

print("Simulated chain:")
print(chain)
```

The important line is:

```python
current_state = np.random.choice(
    n_states,
    p=P[current_state]
)
```

It says: use the transition probabilities of the current state to randomly choose the next state.

---

## 7. Convert State Numbers Back to State Names

```python
chain_states = [
    unique_states[i]
    for i in chain
]

print(chain_states)
```

---

## 8. Count How Often Each State Appears

```python
state_counts = np.bincount(
    chain,
    minlength=n_states
)

for state, count in zip(unique_states, state_counts):
    print(f"{state}: {count} times")
```

---

## 9. Estimate Empirical State Frequencies

Empirical frequency:

    frequency =
    number of times state appears / total chain length

```python
empirical_frequencies = state_counts / len(chain)

for state, frequency in zip(
    unique_states,
    empirical_frequencies
):
    print(f"{state}: {frequency:.3f}")
```

These frequencies should add up to 1.

---

## 10. Plot State Over Time

```python
plt.figure(figsize=(10, 4))

plt.plot(
    range(length),
    chain,
    marker="o",
    markersize=3
)

plt.xlabel("Time")
plt.ylabel("State")
plt.title("Simulated Markov Chain")

plt.show()
```

The x-axis is time and the y-axis is the current state.

---

## 11. State Probabilities After n Steps

If the initial probability distribution is:

    [1, 0, 0]

then we start in state 0 with probability 1.

The probability distribution after `n` steps is:

    pi_n = pi_0 P^n

```python
# Initial distribution
initial = np.zeros(n_states)
initial[0] = 1

# Number of steps
n = 10

# Calculate P^n and multiply by initial distribution
probability_after_n = (
    initial @ np.linalg.matrix_power(P, n)
)

print("State probabilities after", n, "steps:")
print(probability_after_n)
```

---

## 12. Stationary Distribution

A stationary distribution is a probability distribution that does not change after applying the transition matrix.

It satisfies:

    pi P = pi

A simple numerical method is to repeatedly multiply a distribution by `P`.

```python
# Start with uniform distribution
pi = np.ones(n_states) / n_states

# Repeatedly apply P
for _ in range(1000):
    pi = pi @ P

print("Stationary distribution:")

for state, probability in zip(
    unique_states,
    pi
):
    print(f"{state}: {probability:.4f}")
```

Check it:

```python
print("pi:")
print(pi)

print("pi @ P:")
print(pi @ P)
```

They should be approximately equal.

---

## 13. Hitting Time

Hitting time is the number of steps needed to reach a target state.

Example:

    Start = state 0
    Target = state 2

```python
start = 0
target = 2

current = start
time = 0

while current != target:
    current = np.random.choice(
        n_states,
        p=P[current]
    )

    time += 1

print("Hitting time:", time)
```

Because the chain is random, the hitting time can be different each time.

---

## 14. Estimate Expected Hitting Time

Repeat the simulation many times and take the average.

```python
trials = 10000

hitting_times = []

for _ in range(trials):

    current = start
    time = 0

    while current != target and time < 10000:
        current = np.random.choice(
            n_states,
            p=P[current]
        )

        time += 1

    hitting_times.append(time)

expected_hitting_time = np.mean(hitting_times)

print(
    "Estimated expected hitting time:",
    expected_hitting_time
)
```

---

## 15. Long-Run Behavior

To examine long-run behavior, simulate a much longer chain, such as 10,000 steps.

```python
length = 10000

current_state = 0
long_chain = [current_state]

for _ in range(length - 1):
    current_state = np.random.choice(
        n_states,
        p=P[current_state]
    )

    long_chain.append(current_state)

long_run_counts = np.bincount(
    long_chain,
    minlength=n_states
)

long_run_frequencies = (
    long_run_counts / length
)

print("Long-run frequencies:")

for state, frequency in zip(
    unique_states,
    long_run_frequencies
):
    print(f"{state}: {frequency:.4f}")
```

Under suitable conditions such as irreducibility and aperiodicity, the empirical long-run frequencies approach the stationary distribution.

---

# 16. Complete Exam Version

This is the compact version to remember for an exam.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset.csv")

states = df["state"].values

unique_states = np.sort(
    df["state"].unique()
)

n_states = len(unique_states)

print("States:", unique_states)


# ==========================================
# 2. TRANSITION COUNT MATRIX
# ==========================================

counts = np.zeros(
    (n_states, n_states)
)

state_to_index = {
    state: i
    for i, state in enumerate(unique_states)
}

for i in range(len(states) - 1):

    current = state_to_index[states[i]]

    next_state = state_to_index[
        states[i + 1]
    ]

    counts[current, next_state] += 1

print("\nTransition counts:")
print(counts)


# ==========================================
# 3. TRANSITION MATRIX
# ==========================================

P = counts / counts.sum(
    axis=1,
    keepdims=True
)

print("\nTransition matrix:")
print(P)


# ==========================================
# 4. CHECK ROW SUMS
# ==========================================

print("\nRow sums:")
print(P.sum(axis=1))

assert np.allclose(
    P.sum(axis=1),
    1
)

print("Every row sums to 1!")


# ==========================================
# 5. SIMULATE 100 STEPS
# ==========================================

length = 100

current_state = 0
chain = [current_state]

for _ in range(length - 1):

    current_state = np.random.choice(
        n_states,
        p=P[current_state]
    )

    chain.append(current_state)

print("\nSimulated chain:")
print(chain)


# ==========================================
# 6. COUNT STATES
# ==========================================

state_counts = np.bincount(
    chain,
    minlength=n_states
)

print("\nState counts:")

for state, count in zip(
    unique_states,
    state_counts
):
    print(f"{state}: {count}")


# ==========================================
# 7. EMPIRICAL FREQUENCIES
# ==========================================

empirical_frequencies = (
    state_counts / len(chain)
)

print("\nEmpirical frequencies:")

for state, frequency in zip(
    unique_states,
    empirical_frequencies
):
    print(f"{state}: {frequency:.4f}")


# ==========================================
# 8. STATE PROBABILITY AFTER n STEPS
# ==========================================

initial = np.zeros(n_states)
initial[0] = 1

n = 10

probability_after_n = (
    initial
    @ np.linalg.matrix_power(P, n)
)

print(
    "\nProbabilities after",
    n,
    "steps:"
)

print(probability_after_n)


# ==========================================
# 9. STATIONARY DISTRIBUTION
# ==========================================

pi = np.ones(n_states) / n_states

for _ in range(1000):
    pi = pi @ P

print("\nStationary distribution:")

for state, probability in zip(
    unique_states,
    pi
):
    print(f"{state}: {probability:.4f}")


# ==========================================
# 10. HITTING TIME
# ==========================================

start = 0
target = min(2, n_states - 1)

current = start
time = 0

while current != target:

    current = np.random.choice(
        n_states,
        p=P[current]
    )

    time += 1

print(
    "\nHitting time:",
    time
)


# ==========================================
# 11. LONG-RUN BEHAVIOR
# ==========================================

length = 10000

current_state = 0
long_chain = [current_state]

for _ in range(length - 1):

    current_state = np.random.choice(
        n_states,
        p=P[current_state]
    )

    long_chain.append(current_state)

long_run_counts = np.bincount(
    long_chain,
    minlength=n_states
)

long_run_frequencies = (
    long_run_counts / length
)

print("\nLong-run frequencies:")

for state, frequency in zip(
    unique_states,
    long_run_frequencies
):
    print(f"{state}: {frequency:.4f}")


# ==========================================
# 12. PLOT
# ==========================================

plt.figure(figsize=(10, 4))

plt.plot(
    chain,
    marker="o",
    markersize=3
)

plt.xlabel("Time")
plt.ylabel("State")
plt.title("Simulated Markov Chain")

plt.show()
```

---

# 17. Important Formulas

### Transition Probability

    P(i,j) =
    number of transitions i -> j
    --------------------------------
    total transitions from i

### State Probability After n Steps

    pi_n = pi_0 P^n

### Stationary Distribution

    pi P = pi

### Empirical Frequency

    frequency(i) =
    number of visits to state i
    ---------------------------
    total observations

### Hitting Time

Number of transitions required to reach a target state.

---

# 18. The Full Concept

Remember this pipeline:

    Dataset
       |
       v
    Observed states
       |
       v
    Count consecutive transitions
       |
       v
    Transition count matrix
       |
       v
    Normalize each row
       |
       v
    Transition matrix P
       |
       v
    Check row sums = 1
       |
       v
    Simulate Markov chain
       |
       +----> State counts
       |
       +----> Empirical frequencies
       |
       +----> State probabilities
       |
       +----> Stationary distribution
       |
       +----> Hitting time
       |
       +----> Long-run behavior

The three most important pieces of code are:

```python
# Count transitions
for i in range(len(states) - 1):
    current = state_to_index[states[i]]
    next_state = state_to_index[states[i + 1]]
    counts[current, next_state] += 1
```

```python
# Convert counts to probabilities
P = counts / counts.sum(
    axis=1,
    keepdims=True
)
```

```python
# Simulate next state
current_state = np.random.choice(
    n_states,
    p=P[current_state]
)
```

The key idea is:

**Dataset -> count transitions -> normalize rows -> transition matrix -> simulate/analyse the Markov chain.**
