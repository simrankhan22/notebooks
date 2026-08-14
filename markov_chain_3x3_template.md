# Markov Chain — 3×3 Transition Matrix Python Template

This template is for when the **3×3 transition matrix is already given**. You do NOT need a dataset.

It covers:
- Row-sum check
- Simulation of a chain of length 100
- State counts
- Empirical state frequencies
- Plotting
- State probabilities after n steps
- Stationary distribution
- Hitting time
- Expected hitting time
- Long-run behavior

---

## 1. Define the 3×3 Transition Matrix

Rows = current state. Columns = next state.

```python
import numpy as np
import matplotlib.pyplot as plt

P = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.5, 0.3],
    [0.1, 0.2, 0.7]
])

states = [0, 1, 2]
n_states = len(states)
```

If the exam gives a different matrix, **only replace `P`**.

---

## 2. Check That Each Row Sums to 1

A valid transition matrix satisfies:

```text
sum of every row = 1
```

```python
print("Transition matrix:")
print(P)

print("\nRow sums:")
print(P.sum(axis=1))

assert np.allclose(P.sum(axis=1), 1)

print("Every row sums to 1!")
```

---

## 3. Simulate One Markov Chain of Length 100

Start in state 0. At each step, use the current state's row of `P` to randomly select the next state.

```python
length = 100

current_state = 0
chain = [current_state]

for _ in range(length - 1):
    current_state = np.random.choice(
        states,
        p=P[current_state]
    )
    chain.append(current_state)

print("\nSimulated chain:")
print(chain)
```

The key line is:

```python
current_state = np.random.choice(
    states,
    p=P[current_state]
)
```

If the current state is 0, it uses `P[0]` as the probabilities for the next state.

---

## 4. Count How Often Each State Appears

```python
state_counts = np.bincount(
    chain,
    minlength=n_states
)

print("\nState counts:")

for state, count in zip(states, state_counts):
    print(f"State {state}: {count}")
```

---

## 5. Estimate Empirical State Frequencies

Formula:

```text
frequency = number of visits / total chain length
```

```python
frequencies = state_counts / len(chain)

print("\nEmpirical frequencies:")

for state, frequency in zip(states, frequencies):
    print(f"State {state}: {frequency:.4f}")
```

The frequencies should add up to 1.

---

## 6. Plot State Over Time

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

---

## 7. State Probabilities After n Steps

If:

```python
initial = np.array([1, 0, 0])
```

we start in state 0 with probability 1.

Formula:

```text
π_n = π_0 P^n
```

```python
initial = np.array([1, 0, 0])

n = 10

probability_after_n = (
    initial @ np.linalg.matrix_power(P, n)
)

print(f"\nProbabilities after {n} steps:")
print(probability_after_n)
```

If the starting distribution is different, change `initial`.

---

## 8. Stationary Distribution

A stationary distribution satisfies:

```text
πP = π
```

It represents the long-run distribution under suitable conditions.

```python
pi = np.ones(n_states) / n_states

for _ in range(1000):
    pi = pi @ P

print("\nStationary distribution:")

for state, probability in zip(states, pi):
    print(f"State {state}: {probability:.4f}")
```

Check:

```python
print("\npi:")
print(pi)

print("\npi @ P:")
print(pi @ P)
```

These should be approximately equal.

---

## 9. Hitting Time

Hitting time = number of steps needed to reach a target state.

Example: start at state 0 and target state 2.

```python
start = 0
target = 2

current = start
time = 0

while current != target:
    current = np.random.choice(
        states,
        p=P[current]
    )
    time += 1

print(
    f"\nHitting time from state {start} "
    f"to state {target}: {time}"
)
```

Because the process is random, the result can change each run.

---

## 10. Estimate Expected Hitting Time

Repeat the experiment many times and take the average.

```python
trials = 10000
hitting_times = []

for _ in range(trials):
    current = start
    time = 0

    while current != target and time < 10000:
        current = np.random.choice(
            states,
            p=P[current]
        )
        time += 1

    hitting_times.append(time)

expected_hitting_time = np.mean(hitting_times)

print(
    "\nEstimated expected hitting time:",
    expected_hitting_time
)
```

---

## 11. Long-Run Behavior

Simulate a much longer chain and calculate its empirical frequencies.

```python
long_length = 10000

current_state = 0
long_chain = [current_state]

for _ in range(long_length - 1):
    current_state = np.random.choice(
        states,
        p=P[current_state]
    )
    long_chain.append(current_state)

long_counts = np.bincount(
    long_chain,
    minlength=n_states
)

long_frequencies = long_counts / long_length

print("\nLong-run frequencies:")

for state, frequency in zip(
    states,
    long_frequencies
):
    print(f"State {state}: {frequency:.4f}")
```

For a suitable chain (for example, irreducible and aperiodic), the long-run empirical frequencies should approach the stationary distribution.

---

## 12. Compare Stationary and Long-Run Frequencies

```python
print("\nComparison:")

for state, stationary, empirical in zip(
    states,
    pi,
    long_frequencies
):
    print(
        f"State {state}: "
        f"stationary = {stationary:.4f}, "
        f"long-run = {empirical:.4f}"
    )
```

---

# COMPLETE EXAM TEMPLATE

```python
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. TRANSITION MATRIX
# ============================================================

P = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.5, 0.3],
    [0.1, 0.2, 0.7]
])

states = [0, 1, 2]
n_states = len(states)


# ============================================================
# 2. CHECK ROW SUMS
# ============================================================

print("Transition matrix:")
print(P)

print("\nRow sums:")
print(P.sum(axis=1))

assert np.allclose(P.sum(axis=1), 1)

print("Every row sums to 1!")


# ============================================================
# 3. SIMULATE CHAIN OF LENGTH 100
# ============================================================

length = 100
current_state = 0
chain = [current_state]

for _ in range(length - 1):
    current_state = np.random.choice(
        states,
        p=P[current_state]
    )
    chain.append(current_state)

print("\nSimulated chain:")
print(chain)


# ============================================================
# 4. COUNT STATES
# ============================================================

state_counts = np.bincount(
    chain,
    minlength=n_states
)

print("\nState counts:")

for state, count in zip(states, state_counts):
    print(f"State {state}: {count}")


# ============================================================
# 5. EMPIRICAL FREQUENCIES
# ============================================================

frequencies = state_counts / len(chain)

print("\nEmpirical frequencies:")

for state, frequency in zip(states, frequencies):
    print(f"State {state}: {frequency:.4f}")


# ============================================================
# 6. PLOT
# ============================================================

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


# ============================================================
# 7. STATE PROBABILITIES AFTER n STEPS
# ============================================================

initial = np.array([1, 0, 0])
n = 10

probability_after_n = (
    initial @ np.linalg.matrix_power(P, n)
)

print(f"\nProbabilities after {n} steps:")
print(probability_after_n)


# ============================================================
# 8. STATIONARY DISTRIBUTION
# ============================================================

pi = np.ones(n_states) / n_states

for _ in range(1000):
    pi = pi @ P

print("\nStationary distribution:")

for state, probability in zip(states, pi):
    print(f"State {state}: {probability:.4f}")

print("\nCheck pi @ P:")
print(pi @ P)


# ============================================================
# 9. HITTING TIME
# ============================================================

start = 0
target = 2

current = start
time = 0

while current != target:
    current = np.random.choice(
        states,
        p=P[current]
    )
    time += 1

print(
    f"\nHitting time from state {start} "
    f"to state {target}: {time}"
)


# ============================================================
# 10. EXPECTED HITTING TIME
# ============================================================

trials = 10000
hitting_times = []

for _ in range(trials):
    current = start
    time = 0

    while current != target and time < 10000:
        current = np.random.choice(
            states,
            p=P[current]
        )
        time += 1

    hitting_times.append(time)

expected_hitting_time = np.mean(hitting_times)

print(
    "\nEstimated expected hitting time:",
    expected_hitting_time
)


# ============================================================
# 11. LONG-RUN BEHAVIOR
# ============================================================

long_length = 10000

current_state = 0
long_chain = [current_state]

for _ in range(long_length - 1):
    current_state = np.random.choice(
        states,
        p=P[current_state]
    )
    long_chain.append(current_state)

long_counts = np.bincount(
    long_chain,
    minlength=n_states
)

long_frequencies = long_counts / long_length

print("\nLong-run frequencies:")

for state, frequency in zip(
    states,
    long_frequencies
):
    print(f"State {state}: {frequency:.4f}")


# ============================================================
# 12. COMPARE STATIONARY VS LONG-RUN
# ============================================================

print("\nComparison:")

for state, stationary, empirical in zip(
    states,
    pi,
    long_frequencies
):
    print(
        f"State {state}: "
        f"stationary = {stationary:.4f}, "
        f"long-run = {empirical:.4f}"
    )
```

---

# Quick Formula Sheet

### Transition Matrix

```text
P[i,j] = P(X_{t+1}=j | X_t=i)
```

### Row Sum

```text
Σ_j P[i,j] = 1
```

### Probability After n Steps

```text
π_n = π_0 P^n
```

### Stationary Distribution

```text
πP = π
```

### Empirical Frequency

```text
frequency(i) =
visits to state i / total observations
```

### Hitting Time

Number of transitions needed to reach a target state.

### Long-Run Behavior

For a suitable Markov chain:

```text
empirical frequencies → stationary distribution
```

---

# What to Change in the Exam

Usually you only need to change:

### 1. The matrix

```python
P = np.array([
    [YOUR, VALUES, HERE],
    [YOUR, VALUES, HERE],
    [YOUR, VALUES, HERE]
])
```

### 2. Starting state

```python
current_state = 0
```

### 3. Initial distribution

```python
initial = np.array([1, 0, 0])
```

### 4. Number of steps

```python
n = 10
```

### 5. Hitting-time start and target

```python
start = 0
target = 2
```

Everything else can stay the same.

---

# Exam Mental Model

```text
Given P
  ↓
Check rows sum to 1
  ↓
Simulate
  ↓
Count states
  ↓
Empirical frequencies
  ↓
π₀Pⁿ
  ↓
Stationary π
  ↓
Hitting time
  ↓
Long-run frequencies
```

**If the 3×3 transition matrix is already given, you do NOT need a dataset or transition-counting step.**
