# SVD Practice Questions + Copy-Paste Answers
## MK_1MS041 — Based on the supplied exam/template material

These are **question + answer templates**, not theory notes. They deliberately follow the style and task structure of the supplied SVD material while changing the context/task slightly rather than merely changing numbers.

The supplied exam uses:
- `X = U D V^T`
- `np.linalg.svd`
- `Vt = V^T`
- first left/right singular vectors
- explained variance from squared singular values
- choosing the smallest number of components
- low-rank reconstruction
- row-wise reconstruction error
- EDF/outlier detection
- SVD/low-rank approximation exercises
- PCA-style centering before SVD

---

# QUESTION 1 — Compute and Inspect an SVD

### Question

You are given a data matrix `X` with shape `(n_samples, n_dimensions)`.

Compute a compact SVD

\[
X = UDV^T.
\]

Store:

- the left singular-vector matrix in `U`
- the diagonal singular-value matrix in `D`
- the right singular-vector matrix in `V`
- the first left singular vector in `first_left`
- the first right singular vector in `first_right`

Also verify that the SVD reconstructs the original matrix.

### Answer

```python
import numpy as np

# X is already provided by the exam
# X = ...

U, s, Vt = np.linalg.svd(
    X,
    full_matrices=False
)

D = np.diag(s)
V = Vt.T

first_left = U[:, 0].flatten()
first_right = V[:, 0].flatten()

# Reconstruction
X_reconstructed = U @ D @ V.T

reconstruction_error = np.linalg.norm(
    X - X_reconstructed
)

print("U shape:", U.shape)
print("D shape:", D.shape)
print("V shape:", V.shape)
print("reconstruction error:", reconstruction_error)
```

### What you should remember

```text
np.linalg.svd → U, s, Vt
V = Vt.T
D = np.diag(s)
first left  = U[:, 0]
first right = V[:, 0]
```

The supplied exam explicitly warns that NumPy returns `Vt`, which is \(V^T\). It also asks for the first left/right vectors as 1-D arrays. fileciteturn7file3L28-L34 fileciteturn7file3L42-L52

---

# QUESTION 2 — Explained Variance and Number of Components

### Question

Using the singular values obtained from Question 1, calculate the cumulative explained variance

\[
EV(k)=
\frac{\sum_{i=1}^{k}\sigma_i^2}
{\sum_{i=1}^{N}\sigma_i^2}.
\]

Then find the smallest number of components needed to explain at least **98%** of the variance.

Plot the cumulative explained variance.

### Answer

```python
# Squared singular values
s_squared = s**2

# Cumulative explained variance
explained_variance = (
    np.cumsum(s_squared) /
    np.sum(s_squared)
)

# Smallest k reaching 98%
target = 0.98

k = np.argmax(
    explained_variance >= target
) + 1

print("Number of components:", k)
print("Explained variance at k:", explained_variance[k-1])
```

Plot:

```python
import matplotlib.pyplot as plt

plt.plot(
    np.arange(1, len(s) + 1),
    explained_variance,
    marker="o"
)

plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Cumulative Explained Variance")
plt.grid()
plt.show()
```

### Interpretation answer

> The explained variance increases as more singular components are included. Since the singular values are ordered from largest to smallest, the first components contribute the most variance. The selected `k` is the smallest number of components for which the cumulative explained variance reaches at least 98%.

The supplied exam uses exactly this squared-singular-value definition and asks for the smallest `k` reaching a target such as 99%. fileciteturn7file3L36-L40

---

# QUESTION 3 — Reconstruct Using Only the First k Components

### Question

Using the `k` found in Question 2, construct a rank-`k` approximation of `X`.

Call the result `X_k`.

### Answer

```python
X_k = (
    U[:, :k]
    @ np.diag(s[:k])
    @ Vt[:k, :]
)
```

Or, if you have already constructed `V`:

```python
X_k = (
    U[:, :k]
    @ np.diag(s[:k])
    @ V[:, :k].T
)
```

### Why?

The truncated SVD is

\[
X_k=U_kD_kV_k^T.
\]

So:

```text
U[:, :k]  → first k columns
s[:k]     → first k singular values
Vt[:k,:]  → first k rows of V^T
```

The supplied material gives this exact rank-`k` construction. fileciteturn6file1L107-L112

---

# QUESTION 4 — Reconstruction Error

### Question

Calculate the reconstruction error between the original matrix `X` and its rank-`k` approximation `X_k` using the Frobenius norm.

Then calculate the reconstruction error **for every row**.

### Answer

Overall Frobenius error:

```python
frobenius_error = np.linalg.norm(
    X - X_k,
    ord="fro"
)

print(frobenius_error)
```

Row-wise error:

```python
row_error = np.linalg.norm(
    X - X_k,
    axis=1
)

print(row_error)
print(row_error.shape)
```

### Important distinction

```text
Frobenius:
np.linalg.norm(X - X_k, ord="fro")
→ ONE number

Row-wise:
np.linalg.norm(X - X_k, axis=1)
→ ONE number per row
```

The supplied final exam specifically asks for row-wise Euclidean reconstruction errors. fileciteturn6file2L180-L186

---

# QUESTION 5 — SVD-Based Anomaly Detection

### Question

You are told that observations which cannot be well represented by the low-rank structure of the data are potential anomalies.

Using the rank-`k` reconstruction from Question 3:

1. Calculate the reconstruction error for every observation.
2. Identify the 20 observations with the largest reconstruction errors.
3. Store their rows in `outliers`.

### Answer

```python
# Reconstruction error for each observation
row_error = np.linalg.norm(
    X - X_k,
    axis=1
)

# Indices of 20 largest errors
outlier_indices = np.argsort(row_error)[-20:]

# Store outlier rows
outliers = X[outlier_indices]

print("Outlier shape:", outliers.shape)
```

Expected shape:

```text
(20, n_dimensions)
```

### Interpretation

> A large reconstruction error means that the observation is poorly represented by the low-rank structure learned from the dominant singular components. Such observations can therefore be treated as potential outliers.

The supplied exam uses this same low-rank-reconstruction-error idea for anomaly detection. fileciteturn6file0L34-L39

---

# QUESTION 6 — Exactly 100 Outliers + Threshold

### Question

Using the reconstruction errors, choose a threshold so that the observations with the largest reconstruction errors are considered outliers. The task requires **exactly 100 outlier observations**.

Store:

- the threshold in `threshold`
- the selected observations in `outliers`

### Answer

```python
n_outliers = 100

# Indices of the 100 largest errors
outlier_indices = np.argsort(row_error)[-n_outliers:]

# Threshold = smallest error among selected outliers
threshold = np.min(
    row_error[outlier_indices]
)

outliers = X[outlier_indices]

print("threshold:", threshold)
print("number of outliers:", len(outliers))
```

### Important

Do not blindly use:

```python
outlier_indices = np.where(row_error >= threshold)[0]
```

if the question requires **exactly** 100 observations, because ties at the threshold can produce more than 100.

The supplied final exam specifically asks for a threshold such that exactly 100 samples are flagged. fileciteturn6file0L34-L39

---

# QUESTION 7 — EDF of Reconstruction Errors

### Question

Plot the empirical distribution function (EDF) of the row-wise reconstruction errors.

### Answer

```python
sorted_errors = np.sort(row_error)

edf_y = (
    np.arange(1, len(sorted_errors) + 1)
    / len(sorted_errors)
)

import matplotlib.pyplot as plt

plt.step(
    sorted_errors,
    edf_y,
    where="post"
)

plt.xlabel("Reconstruction error")
plt.ylabel("EDF")
plt.title("EDF of Reconstruction Errors")
plt.grid()
plt.show()
```

### Interpretation

The EDF shows the proportion of observations having reconstruction error less than or equal to a given value.

---

# QUESTION 8 — SVD Reconstruction Without Truncation

### Question

A matrix `A` of shape `(50, 30)` is generated randomly.

1. Compute its SVD.
2. Reconstruct the complete matrix using all singular components.
3. Calculate the reconstruction error.

### Answer

```python
import numpy as np

A = np.random.randn(50, 30)

U, s, Vt = np.linalg.svd(
    A,
    full_matrices=False
)

A_reconstructed = (
    U @ np.diag(s) @ Vt
)

error = np.linalg.norm(
    A - A_reconstructed,
    ord="fro"
)

print("A shape:", A.shape)
print("Reconstruction error:", error)
```

Expected result:

```text
reconstruction error ≈ 0
```

The supplied course template explicitly includes the task of creating a `(50, 30)` matrix, computing SVD, and reconstructing it. fileciteturn7file0L10-L15

---

# QUESTION 9 — Compare Several Ranks

### Question

For a matrix `A`, construct rank-1, rank-2, ..., rank-`r` approximations.

Calculate the Frobenius reconstruction error for each rank and plot error against rank.

### Answer

```python
U, s, Vt = np.linalg.svd(
    A,
    full_matrices=False
)

errors = []

for k in range(1, len(s) + 1):

    A_k = (
        U[:, :k]
        @ np.diag(s[:k])
        @ Vt[:k, :]
    )

    error = np.linalg.norm(
        A - A_k,
        ord="fro"
    )

    errors.append(error)
```

Plot:

```python
ranks = np.arange(1, len(s) + 1)

plt.plot(
    ranks,
    errors,
    marker="o"
)

plt.xlabel("Rank")
plt.ylabel("Reconstruction error")
plt.title("Low-Rank Approximation Error")
plt.grid()
plt.show()
```

### Expected interpretation

> The reconstruction error decreases as the rank increases because more singular components are retained.

---

# QUESTION 10 — Explain Why Low-Rank Approximation Loses Information

### Question

Explain why a rank-`k` approximation generally contains less information than the original matrix.

### Answer

> The SVD represents the matrix using singular components ordered by singular value. A rank-`k` approximation keeps only the first `k` components and discards the remaining components. Since the discarded singular values and vectors contribute to the original matrix, some information is lost.

This matches the supplied Day 7 explanation: low-rank approximation discards smaller singular values and corresponding singular vectors, so some information is lost. fileciteturn6file2L188-L195

---

# QUESTION 11 — Explain Why the First Components Matter Most

### Question

Why does keeping the first few singular components often preserve most of the information?

### Answer

> The singular values are ordered from largest to smallest. The squared singular values determine the contribution to the explained variance, so the first singular components account for the largest amount of variance. Therefore, keeping the first few components can preserve most of the important structure while reducing the dimensionality.

---

# QUESTION 12 — Find the Rank from the Singular Values

### Question

Suppose the SVD of a matrix produces singular values

```text
[12.4, 7.1, 3.2, 0.0, 0.0]
```

What is the rank?

Then explain how you would determine the numerical rank in Python.

### Answer

The rank is:

```text
3
```

because there are three non-zero singular values.

Python:

```python
rank = np.linalg.matrix_rank(A)

print(rank)
```

Conceptually:

\[
rank(A)=\#\{\sigma_i\neq0\}.
\]

---

# QUESTION 13 — PCA/SVD with Centering

### Question

You are given a feature matrix `X`. Before applying SVD, center each feature by subtracting its column mean.

Then compute the SVD of the centered matrix.

### Answer

```python
# Column-wise centering
X_centered = X - X.mean(axis=0)

# SVD
U, s, Vt = np.linalg.svd(
    X_centered,
    full_matrices=False
)

D = np.diag(s)
V = Vt.T
```

### Important

This is **column-wise** centering:

```python
X.mean(axis=0)
```

not:

```python
X.mean(axis=1)
```

---

# QUESTION 14 — PCA: Choose Enough Components for 95%

### Question

After centering `X` and computing its SVD, find the smallest number of components that explain at least 95% of the variance.

### Answer

```python
explained_variance = (
    np.cumsum(s**2)
    / np.sum(s**2)
)

target = 0.95

n_components = (
    np.argmax(
        explained_variance >= target
    ) + 1
)

print(n_components)
print(
    explained_variance[n_components - 1]
)
```

---

# QUESTION 15 — PCA: Two-Dimensional Representation

### Question

Using the centered data and its SVD, create a two-dimensional representation of the observations using the first two principal components.

### Answer

```python
X_2d = (
    U[:, :2]
    @ np.diag(s[:2])
)
```

Equivalent:

```python
X_2d = X_centered @ V[:, :2]
```

The result has shape:

```text
(n_samples, 2)
```

---

# QUESTION 16 — Reconstruct the Original PCA Data

### Question

Using the first `k` principal components, reconstruct the centered data. Then transform the reconstruction back to the original scale.

### Answer

```python
X_centered_k = (
    U[:, :k]
    @ np.diag(s[:k])
    @ Vt[:k, :]
)

# Return to original scale
X_k = X_centered_k + X.mean(axis=0)
```

Remember:

```text
X_centered_k → reconstruction of centered data
X_k          → reconstruction on original scale
```

---

# QUESTION 17 — Combined Exam-Style SVD Problem

## Question

A matrix `X` is provided as a NumPy array with shape `(n_samples, n_dimensions)`.

### Part A — SVD [4p]

Compute a compact SVD

\[
X=UDV^T.
\]

Store `U`, `D`, and `V`.

Also extract the first left and right singular vectors.

### Part B — Explained variance [3p]

Calculate the cumulative explained variance for all components.

Find the smallest `k` that explains at least 99% of the variance.

### Part C — Low-rank reconstruction [4p]

Construct the rank-`k` approximation `X_k`.

Calculate the row-wise reconstruction error.

### Part D — Outliers [3p]

Plot the EDF of the reconstruction errors and identify the 50 observations with the largest reconstruction errors.

---

## Answer

### Part A

```python
U, s, Vt = np.linalg.svd(
    X,
    full_matrices=False
)

D = np.diag(s)
V = Vt.T

first_left = U[:, 0].flatten()
first_right = V[:, 0].flatten()
```

### Part B

```python
explained_variance = (
    np.cumsum(s**2)
    / np.sum(s**2)
)

k = np.argmax(
    explained_variance >= 0.99
) + 1

print("k =", k)
```

### Part C

```python
X_k = (
    U[:, :k]
    @ np.diag(s[:k])
    @ Vt[:k, :]
)

row_error = np.linalg.norm(
    X - X_k,
    axis=1
)
```

### Part D

```python
sorted_errors = np.sort(row_error)

edf_y = (
    np.arange(1, len(sorted_errors) + 1)
    / len(sorted_errors)
)

plt.step(
    sorted_errors,
    edf_y,
    where="post"
)

plt.xlabel("Reconstruction error")
plt.ylabel("EDF")
plt.grid()
plt.show()
```

Top 50:

```python
n_outliers = 50

outlier_indices = np.argsort(
    row_error
)[-n_outliers:]

outliers = X[outlier_indices]
```

---

# QUESTION 18 — Debug This SVD Code

### Question

A student writes:

```python
U, s, V = np.linalg.svd(A, full_matrices=False)

D = np.diag(s)

A_k = U[:, :k] @ D[:k, :] @ V[:k, :]
```

The result has an unexpected shape/error.

What is wrong? Rewrite the code correctly.

### Answer

The main issue is the interpretation of NumPy's third output.

NumPy returns:

```python
U, s, Vt
```

where the third object is \(V^T\), not \(V\).

Correct:

```python
U, s, Vt = np.linalg.svd(
    A,
    full_matrices=False
)

A_k = (
    U[:, :k]
    @ np.diag(s[:k])
    @ Vt[:k, :]
)
```

If you actually need `V`:

```python
V = Vt.T
```

---

# QUESTION 19 — Debug Explained Variance

### Question

A student calculates:

```python
ev = np.cumsum(s) / np.sum(s)
```

The question defines explained variance using squared singular values.

Correct the code.

### Answer

```python
ev = np.cumsum(s**2) / np.sum(s**2)
```

Then:

```python
k = np.argmax(ev >= 0.99) + 1
```

The supplied exam explicitly defines explained variance using \(\sigma_i^2\). fileciteturn7file3L36-L40

---

# QUESTION 20 — Debug Reconstruction Error

### Question

The exam asks:

> Calculate the reconstruction error of each observation.

A student writes:

```python
error = np.linalg.norm(
    X - X_k,
    ord="fro"
)
```

Is this the correct answer?

### Answer

**No**, not if the exam wants one error for every observation.

`ord="fro"` produces one total matrix error.

Correct:

```python
error = np.linalg.norm(
    X - X_k,
    axis=1
)
```

This returns one Euclidean reconstruction error per row.

---

# QUICK COPY-PASTE ANSWER BANK

## Compute SVD

```python
U, s, Vt = np.linalg.svd(
    X,
    full_matrices=False
)
```

## Construct D

```python
D = np.diag(s)
```

## Construct V

```python
V = Vt.T
```

## First left vector

```python
first_left = U[:, 0].flatten()
```

## First right vector

```python
first_right = V[:, 0].flatten()
```

## Explained variance

```python
ev = np.cumsum(s**2) / np.sum(s**2)
```

## Smallest k for target

```python
k = np.argmax(ev >= target) + 1
```

## Rank-k approximation

```python
X_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

## Total reconstruction error

```python
error = np.linalg.norm(
    X - X_k,
    ord="fro"
)
```

## Row-wise reconstruction error

```python
row_error = np.linalg.norm(
    X - X_k,
    axis=1
)
```

## Top N outliers

```python
outlier_indices = np.argsort(row_error)[-N:]
outliers = X[outlier_indices]
```

## Column-center before SVD

```python
X_centered = X - X.mean(axis=0)
```

## PCA/SVD

```python
X_centered = X - X.mean(axis=0)

U, s, Vt = np.linalg.svd(
    X_centered,
    full_matrices=False
)
```

## Full reconstruction check

```python
X_full = U @ np.diag(s) @ Vt

print(
    np.linalg.norm(X - X_full)
)
```

---

# FINAL 10-SECOND MEMORY RULE

When you see an SVD question, immediately write:

```python
U, s, Vt = np.linalg.svd(X, full_matrices=False)
```

Then:

```text
D        = diag(s)
V        = Vt.T
left     = U[:,0]
right    = V[:,0]

EV       = cumsum(s²) / sum(s²)

k        = first index where EV >= target

X_k      = U[:,:k] diag(s[:k]) Vt[:k,:]

total    = norm(X-X_k, 'fro')
row      = norm(X-X_k, axis=1)
```

That sequence covers the main SVD question patterns in the supplied exam/template material.
