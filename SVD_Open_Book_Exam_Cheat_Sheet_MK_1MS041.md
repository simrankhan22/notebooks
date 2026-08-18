# SVD Open-Book Exam Cheat Sheet — MK_1MS041

## Purpose

These notes are strictly focused on the SVD/PCA-style questions appearing in the supplied previous/final exam material.

The goal is **copy-paste smartly**: identify the question type, paste the matching block, and change the matrix/path/target `k` as required.

The supplied exam material covers:
- computing SVD
- singular values
- left/right singular vectors
- `V` versus `Vt`
- explained variance
- selecting the smallest number of components
- low-rank/rank-k approximation
- reconstruction
- reconstruction error
- anomaly/outlier detection
- PCA via centered data
- rank and SVD interpretation

---

# 1. THE CORE SVD RECIPE

For

\[
A = U\Sigma V^T
\]

NumPy gives:

```python
U, s, Vt = np.linalg.svd(A, full_matrices=False)
```

where:

```text
U   = left singular vectors
s   = singular values
Vt  = V^T
```

If the question wants the diagonal matrix:

```python
D = np.diag(s)
```

If the question wants `V`:

```python
V = Vt.T
```

Reconstruction:

```python
A_reconstructed = U @ np.diag(s) @ Vt
```

---

# 2. UNIVERSAL COPY-PASTE TEMPLATE

```python
import numpy as np
import matplotlib.pyplot as plt

# A = your matrix
A = ...

# SVD
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# Diagonal singular-value matrix
D = np.diag(s)

# V rather than V^T
V = Vt.T

print("A:", A.shape)
print("U:", U.shape)
print("s:", s.shape)
print("D:", D.shape)
print("Vt:", Vt.shape)
print("V:", V.shape)
print("singular values:", s)
```

---

# 3. SHAPES

If:

```python
A.shape == (m, n)
```

and:

```python
U, s, Vt = np.linalg.svd(A, full_matrices=False)
```

then with

\[
r=\min(m,n)
\]

you get:

```text
U.shape  = (m, r)
s.shape  = (r,)
Vt.shape = (r, n)
```

If `m >= n`:

```text
U  = m × n
D  = n × n
V  = n × n
```

This is the shape convention used in the supplied exam material for data matrices.

---

# 4. LOAD CSV + SVD

No header:

```python
A = np.loadtxt("data/SVD.csv", delimiter=",")

U, s, Vt = np.linalg.svd(A, full_matrices=False)

D = np.diag(s)
V = Vt.T
```

With a header:

```python
import pandas as pd

df = pd.read_csv("data/SVD.csv")
A = df.to_numpy(dtype=float)

U, s, Vt = np.linalg.svd(A, full_matrices=False)

D = np.diag(s)
V = Vt.T
```

---

# 5. FIRST LEFT / RIGHT SINGULAR VECTORS

First **left** singular vector:

```python
u1 = U[:, 0].flatten()
```

First **right** singular vector:

```python
v1 = Vt[0, :].flatten()
```

Equivalent right-vector form:

```python
v1 = V[:, 0].flatten()
```

Remember:

```text
left  → U[:, 0]
right → Vt[0, :]
```

---

# 6. SINGULAR VALUES

The singular values are already:

```python
s
```

They are ordered:

\[
\sigma_1 \ge \sigma_2 \ge \cdots \ge 0.
\]

Diagonal matrix:

```python
D = np.diag(s)
```

Recover singular values from `D`:

```python
s = np.diag(D)
```

---

# 7. CHECK THE SVD

```python
A_reconstructed = U @ np.diag(s) @ Vt

error = np.linalg.norm(A - A_reconstructed)

print(error)
```

The error should be approximately zero, up to floating-point precision.

---

# 8. EXPLAINED VARIANCE

The supplied exam defines:

\[
EV(k)=
\frac{\sum_{i=1}^{k}\sigma_i^2}
{\sum_{i=1}^{N}\sigma_i^2}.
\]

Copy-paste:

```python
explained_variance = np.cumsum(s**2) / np.sum(s**2)
```

This gives:

```text
EV(1), EV(2), ..., EV(N)
```

The final value should be approximately `1`.

**Important:** use `s**2`, not `s`.

Wrong:

```python
np.cumsum(s) / np.sum(s)
```

Correct:

```python
np.cumsum(s**2) / np.sum(s**2)
```

---

# 9. FIND THE SMALLEST k FOR A TARGET

For 99%:

```python
target = 0.99

k = np.argmax(explained_variance >= target) + 1

print(k)
```

For 95%:

```python
target = 0.95
k = np.argmax(explained_variance >= target) + 1
```

For 90%:

```python
target = 0.90
k = np.argmax(explained_variance >= target) + 1
```

The `+1` is because Python indexing starts at zero.

---

# 10. PLOT EXPLAINED VARIANCE

```python
plt.plot(
    np.arange(1, len(s) + 1),
    explained_variance,
    marker="o"
)

plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Explained Variance")
plt.grid()
plt.show()
```

Interpretation if the curve rises quickly:

> The first few components capture most of the variance, so the data can be represented using relatively few dimensions.

If it rises slowly:

> The variance is distributed across more components, so more components are required to retain most of the information.

---

# 11. RANK-k APPROXIMATION

The key formula is:

\[
A_k=U_k\Sigma_kV_k^T.
\]

Copy-paste:

```python
k = 5

A_k = (
    U[:, :k]
    @ np.diag(s[:k])
    @ Vt[:k, :]
)
```

The compact one-liner:

```python
A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

Remember:

```text
U       → first k COLUMNS
s       → first k VALUES
Vt      → first k ROWS
```

---

# 12. WHY RANK-k WORKS

SVD can be written:

\[
A =
\sigma_1u_1v_1^T+
\sigma_2u_2v_2^T+
\cdots+
\sigma_ru_rv_r^T.
\]

Rank-k approximation keeps:

\[
A_k =
\sum_{i=1}^{k}\sigma_i u_i v_i^T.
\]

It discards the remaining singular components.

So:

```text
large singular values → important components → keep
small singular values → smaller contribution → discard for compression
```

---

# 13. TOTAL RECONSTRUCTION ERROR

If asked for one overall error:

```python
error = np.linalg.norm(A - A_k, ord="fro")
```

This is:

\[
\|A-A_k\|_F.
\]

---

# 14. ROW-WISE RECONSTRUCTION ERROR

If asked for one error **per observation/row**:

```python
reconstruction_error = np.linalg.norm(A - A_k, axis=1)
```

Result:

```text
shape = (number of rows,)
```

Each value is:

\[
\|A_i-(A_k)_i\|_2.
\]

### Remember

```text
ord="fro" → one total matrix error
axis=1    → one error for every row
```

---

# 15. LOW-RANK APPROXIMATION + OUTLIERS

The supplied final exam uses reconstruction error for anomaly detection.

Full template:

```python
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# choose k
k = ...

# rank-k approximation
A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# error for each row
reconstruction_error = np.linalg.norm(
    A - A_k,
    axis=1
)
```

Interpretation:

> An observation that is poorly reconstructed by the low-rank representation has a large reconstruction error and can be treated as a potential outlier.

---

# 16. CHOOSE k FROM EXPLAINED VARIANCE, THEN DETECT OUTLIERS

```python
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# explained variance
explained_variance = np.cumsum(s**2) / np.sum(s**2)

# choose smallest k reaching target
target = 0.99
k = np.argmax(explained_variance >= target) + 1

# rank-k approximation
A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# row-wise error
reconstruction_error = np.linalg.norm(
    A - A_k,
    axis=1
)
```

---

# 17. EXACTLY 100 OUTLIERS

If the question asks for the 100 observations with the largest reconstruction error:

```python
n_outliers = 100

outlier_indices = np.argsort(reconstruction_error)[-n_outliers:]

outliers = A[outlier_indices]
```

Check:

```python
print(outliers.shape)
```

If a threshold is requested:

```python
threshold = np.min(reconstruction_error[outlier_indices])

print(threshold)
```

Be careful with ties: a boolean condition such as `error >= threshold` can select more than 100 observations if several errors are identical.

---

# 18. EDF OF RECONSTRUCTION ERRORS

If asked to plot the empirical distribution function:

```python
sorted_errors = np.sort(reconstruction_error)

y = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

plt.step(sorted_errors, y, where="post")
plt.xlabel("Reconstruction error")
plt.ylabel("EDF")
plt.title("EDF of Reconstruction Errors")
plt.grid()
plt.show()
```

If the exam provides helper functions such as `makeEDF` / `plotEDF`, use those when specifically requested.

---

# 19. ERROR AS A FUNCTION OF RANK

If asked to test several ranks:

```python
errors = []

for k in range(1, len(s) + 1):
    A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    error = np.linalg.norm(A - A_k, ord="fro")
    errors.append(error)
```

Plot:

```python
ranks = np.arange(1, len(s) + 1)

plt.plot(ranks, errors, marker="o")
plt.xlabel("Rank")
plt.ylabel("Reconstruction error")
plt.title("Low-rank approximation error")
plt.grid()
plt.show()
```

Interpretation:

> Reconstruction error decreases as rank increases because more singular components are retained.

---

# 20. MATRIX RANK FROM SVD

Conceptually:

\[
rank(A)=\text{number of non-zero singular values}.
\]

For numerical rank:

```python
rank = np.linalg.matrix_rank(A)
```

Inspect the singular values:

```python
print(s)
```

Large values represent active directions; values effectively equal to zero indicate missing rank directions.

---

# 21. ORTHONORMALITY CHECK

Columns of `U` and `V` are orthonormal.

```python
print(np.allclose(U.T @ U, np.eye(U.shape[1])))
print(np.allclose(V.T @ V, np.eye(V.shape[1])))
```

Mathematically:

\[
U^TU=I,
\]

\[
V^TV=I.
\]

Usually only do this if the question explicitly asks you to verify orthonormality.

---

# 22. PCA/SVD — CENTER THE DATA FIRST

The supplied solved exam includes PCA/SVD where the feature matrix is centered column-wise.

```python
X_centered = X - X.mean(axis=0)

U, s, Vt = np.linalg.svd(
    X_centered,
    full_matrices=False
)

D = np.diag(s)
V = Vt.T
```

Do **not** center rows if the question says column-wise centering.

Correct:

```python
X - X.mean(axis=0)
```

---

# 23. DATASET WITH FEATURES + LABEL

If a dataset contains features followed by a target/label:

```python
data = pd.read_csv("data/digits.csv")

X = data.iloc[:, :64].to_numpy(dtype=float)
y = data.iloc[:, 64].to_numpy()
```

Then:

```python
X_centered = X - X.mean(axis=0)

U, s, Vt = np.linalg.svd(
    X_centered,
    full_matrices=False
)

D = np.diag(s)
V = Vt.T
```

Do not include the target column in the SVD feature matrix when the exam specifies that it is a label.

---

# 24. PCA REDUCED REPRESENTATION

If the question asks for a `k`-dimensional representation:

```python
k = 2

X_reduced = U[:, :k] @ np.diag(s[:k])
```

Equivalent:

```python
X_reduced = X_centered @ V[:, :k]
```

because:

\[
X_cV_k=U_k\Sigma_k.
\]

---

# 25. RECONSTRUCT CENTERED PCA DATA

```python
X_centered_k = (
    U[:, :k]
    @ np.diag(s[:k])
    @ Vt[:k, :]
)
```

If the original scale is needed:

```python
X_k = X_centered_k + X.mean(axis=0)
```

Remember:

```text
X_centered_k = reconstruction on centered scale
X_k          = reconstruction on original scale
```

---

# 26. IMAGE/DIGIT COMPRESSION

For image rows represented by pixels/features:

```python
k = 10

X_k = (
    U[:, :k]
    @ np.diag(s[:k])
    @ Vt[:k, :]
)
```

Interpretation:

> Smaller `k` gives stronger compression but generally loses more information. Larger `k` preserves more information and produces a more accurate reconstruction.

---

# 27. WHY DOES LOW-RANK APPROXIMATION LOSE INFORMATION?

Copy-paste answer:

> Low-rank approximation keeps only the largest singular values and their corresponding singular vectors. The smaller singular components are discarded, so the approximation cannot reproduce all details of the original matrix.

---

# 28. WHY DOES ERROR DECREASE AS k INCREASES?

Copy-paste:

> As \(k\) increases, more singular components are included in the approximation. Therefore, less information from the original matrix is discarded, so the reconstruction becomes more accurate and the reconstruction error decreases.

---

# 29. WHAT DOES SVD DO?

Copy-paste:

> SVD decomposes a matrix into left singular vectors \(U\), singular values \(\Sigma\), and right singular vectors \(V\), such that \(A=U\Sigma V^T\).

---

# 30. WHAT DO THE SINGULAR VALUES MEAN?

Copy-paste:

> The singular values determine the strength or importance of the corresponding singular components. They are ordered from largest to smallest, so the first components generally capture the largest amount of variation/information.

---

# 31. WHAT IS THE ROLE OF U?

> The columns of \(U\) are the left singular vectors. They form orthonormal directions associated with the output/sample space of the matrix.

---

# 32. WHAT IS THE ROLE OF V?

> The columns of \(V\) are the right singular vectors. They form orthonormal directions associated with the input/feature space.

NumPy returns `Vt`, so:

```python
V = Vt.T
```

---

# 33. BEST RANK-k APPROXIMATION

If the question says:

> Find/use the best rank-k approximation.

Use:

```python
A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

Explanation:

> The truncated SVD keeps the \(k\) largest singular values and their corresponding singular vectors. It gives the best rank-\(k\) approximation in the standard Frobenius-norm sense.

---

# 34. IMPORTANT DIFFERENCE: V AND Vt

NumPy:

```python
U, s, Vt = np.linalg.svd(A)
```

Mathematical notation:

\[
A=U\Sigma V^T.
\]

Therefore:

```text
U              → U
s              → singular values
np.diag(s)     → Sigma
Vt             → V^T
Vt.T           → V
```

Common mistake:

```python
# WRONG for reconstruction
A_k = U[:, :k] @ np.diag(s[:k]) @ V[:, :k]
```

Correct:

```python
A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

---

# 35. THE MOST COMMON EXAM TRAPS

### Trap 1 — using `V` when NumPy gave `Vt`

```python
V = Vt.T
```

### Trap 2 — forgetting the diagonal matrix

Wrong:

```python
U @ s @ Vt
```

Correct:

```python
U @ np.diag(s) @ Vt
```

### Trap 3 — wrong rank-k slices

Correct:

```python
U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

### Trap 4 — wrong explained variance

Correct:

```python
np.cumsum(s**2) / np.sum(s**2)
```

### Trap 5 — wrong reconstruction error

Overall:

```python
np.linalg.norm(A - A_k, ord="fro")
```

Per row:

```python
np.linalg.norm(A - A_k, axis=1)
```

### Trap 6 — forgetting PCA centering

```python
X_centered = X - X.mean(axis=0)
```

before SVD if the question says to center.

### Trap 7 — putting labels into PCA/SVD

Use only the feature matrix when the target/label is separate.

---

# 36. DEBUGGING

If dimensions look wrong:

```python
print("A:", A.shape)
print("U:", U.shape)
print("s:", s.shape)
print("Vt:", Vt.shape)
```

If you need `V`:

```python
V = Vt.T
```

If you need the diagonal matrix:

```python
D = np.diag(s)
```

If you need the first vectors:

```python
u1 = U[:, 0]
v1 = Vt[0, :]
```

---

# 37. COMPLETE "ALMOST ANY SVD QUESTION" BLOCK

```python
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# LOAD / CREATE MATRIX
# --------------------------------------------------

A = ...

# --------------------------------------------------
# SVD
# --------------------------------------------------

U, s, Vt = np.linalg.svd(
    A,
    full_matrices=False
)

D = np.diag(s)
V = Vt.T

# --------------------------------------------------
# FIRST SINGULAR VECTORS
# --------------------------------------------------

u1 = U[:, 0].flatten()
v1 = Vt[0, :].flatten()

# --------------------------------------------------
# EXPLAINED VARIANCE
# --------------------------------------------------

ev = np.cumsum(s**2) / np.sum(s**2)

# --------------------------------------------------
# CHOOSE NUMBER OF COMPONENTS
# --------------------------------------------------

target = 0.99
k = np.argmax(ev >= target) + 1

# --------------------------------------------------
# RANK-k APPROXIMATION
# --------------------------------------------------

A_k = (
    U[:, :k]
    @ np.diag(s[:k])
    @ Vt[:k, :]
)

# --------------------------------------------------
# ERRORS
# --------------------------------------------------

fro_error = np.linalg.norm(
    A - A_k,
    ord="fro"
)

row_error = np.linalg.norm(
    A - A_k,
    axis=1
)

# --------------------------------------------------
# CHECK FULL RECONSTRUCTION
# --------------------------------------------------

A_full = U @ np.diag(s) @ Vt

full_error = np.linalg.norm(
    A - A_full
)

print("singular values:", s)
print("number of components:", k)
print("Frobenius reconstruction error:", fro_error)
print("full reconstruction error:", full_error)
```

---

# 38. QUESTION → CODE MAP

| Question wording | Paste this |
|---|---|
| Compute SVD | `U, s, Vt = np.linalg.svd(A, full_matrices=False)` |
| Diagonal matrix | `D = np.diag(s)` |
| Need V | `V = Vt.T` |
| First left vector | `U[:, 0]` |
| First right vector | `Vt[0, :]` |
| Singular values | `s` |
| Explained variance | `np.cumsum(s**2) / np.sum(s**2)` |
| Smallest k for target | `np.argmax(ev >= target) + 1` |
| Rank-k approximation | `U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]` |
| Total error | `np.linalg.norm(A-A_k, ord="fro")` |
| Error per row | `np.linalg.norm(A-A_k, axis=1)` |
| Numerical rank | `np.linalg.matrix_rank(A)` |
| Center columns | `X - X.mean(axis=0)` |
| PCA/SVD | center → `np.linalg.svd(..., full_matrices=False)` |
| Error vs rank | loop over `k`, calculate Frobenius error |
| Outliers | largest row-wise reconstruction errors |

---

# 39. ULTRA-SHORT MEMORY CARD

If you are in a hurry, memorize this:

```python
# SVD
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# D and V
D = np.diag(s)
V = Vt.T

# First left/right vectors
u1 = U[:, 0]
v1 = Vt[0, :]

# Explained variance
ev = np.cumsum(s**2) / np.sum(s**2)

# Smallest k reaching target
k = np.argmax(ev >= target) + 1

# Rank-k approximation
A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# Total reconstruction error
error = np.linalg.norm(A - A_k, ord="fro")

# Row-wise reconstruction error
row_error = np.linalg.norm(A - A_k, axis=1)
```

---

# 40. FINAL DECISION TREE

```text
SVD QUESTION
     |
     v
Compute U, s, Vt
     |
     +---- Need V? --------> V = Vt.T
     |
     +---- Need first left? -> U[:, 0]
     |
     +---- Need first right? -> Vt[0, :]
     |
     +---- Explained variance?
     |          |
     |          v
     |     ev = cumsum(s²)/sum(s²)
     |          |
     |          v
     |     Need k for target?
     |          |
     |          v
     |     k = argmax(ev >= target) + 1
     |
     +---- Reconstruction?
     |          |
     |          v
     |     A_k = U[:,:k] diag(s[:k]) Vt[:k,:]
     |
     +---- Error?
                |
                +--> total: norm(A-A_k, 'fro')
                |
                +--> per row: norm(A-A_k, axis=1)
```

---

# 41. FORMULAS TO KNOW

\[
\boxed{A=U\Sigma V^T}
\]

\[
\boxed{
EV(k)=
\frac{\sum_{i=1}^{k}\sigma_i^2}
{\sum_{i=1}^{r}\sigma_i^2}
}
\]

\[
\boxed{
A_k=U_k\Sigma_kV_k^T
}
\]

\[
\boxed{
A_k=\sum_{i=1}^{k}\sigma_i u_i v_i^T
}
\]

\[
\boxed{
rank(A)=\#\{\sigma_i\neq0\}
}
\]

\[
\boxed{
\text{row error}_i=\|A_i-(A_k)_i\|_2
}
\]

\[
\boxed{
\text{total error}=\|A-A_k\|_F
}
\]

---

# 42. SOURCE-BASED COVERAGE

The supplied January 2026 exam explicitly uses an SVD problem involving:

1. `X = UDV^T`
2. singular values
3. first right and left singular vectors
4. explained variance
5. the smallest number of components reaching 99%
6. explained-variance plotting and interpretation
7. low-rank reconstruction
8. row-wise reconstruction error
9. EDF of reconstruction errors
10. selecting exactly 100 outliers

The supplied solved exam additionally uses:

1. column-wise centering
2. compact SVD with `full_matrices=False`
3. PCA/SVD on digit features
4. cumulative explained variance

The supplied Day 7 material also explicitly covers SVD, matrix rank, and low-rank approximation.

This cheat sheet is organized around those exact question patterns rather than unrelated SVD theory.
