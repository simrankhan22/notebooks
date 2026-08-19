# 1MS041 Data Science --- SVD-Only Questions & Answers

> **Purpose:** This file extracts and reorganizes the SVD-related
> material from the uploaded course preparation/exam files into a single
> study document. It keeps the terminology, formulas, code patterns,
> questions, and supplied answers from the sources. Unrelated topics
> have been omitted.

## 1. SVD fundamentals

### Q1. What does SVD decompose a matrix into?

**Answer:** SVD decomposes a matrix into: - left singular vectors `U` -
singular values `D` (or the 1D singular-value array `S`/`d`) - right
singular vectors `V`

The decomposition is written as

$$
X = UDV^T.
$$

In NumPy, `np.linalg.svd` returns `U, d, Vt`, where `Vt` is $V^T$.

------------------------------------------------------------------------

### Q2. What is the standard SVD notation used in the exam?

**Answer:**

$$
X = UDV^T.
$$

For the January 2026 exam, if `X` has shape

`n_samples × n_dimensions`,

then the requested matrices were: - `U`: `n_samples × n_dimensions` -
`D`: `n_dimensions × n_dimensions` - `V`: `n_dimensions × n_dimensions`

The diagonal entries of `D` are the singular values.

------------------------------------------------------------------------

### Q3. What does `np.linalg.svd` actually return?

**Answer:**

``` python
U, d, Vt = np.linalg.svd(X)
```

where: - `U` contains the left singular vectors, - `d` is a 1D array
containing the singular values, - `Vt` is $V^T$.

To construct the diagonal matrix and $V$:

``` python
D = np.diag(d)
V = Vt.T
```

Therefore:

``` python
X == U @ D @ V.T
```

up to numerical precision.

------------------------------------------------------------------------

### Q4. How do you compute a compact SVD in NumPy?

**Answer:**

``` python
U, singular_values, Vt = np.linalg.svd(
    X,
    full_matrices=False
)
```

The uploaded Exam-2 notebook explicitly emphasizes that **compact SVD
means `full_matrices=False`**.

For a feature matrix with shape `(n_samples, 64)`, the compact shapes
are:

``` text
U.shape == (n_samples, 64)
D.shape == (64, 64)
V.shape == (64, 64)
```

------------------------------------------------------------------------

### Q5. Why is `full_matrices=False` important for the compact SVD?

**Answer:** The notebook's local checks specifically warn that NumPy
uses `full_matrices=True` by default. For the requested compact SVD,
use:

``` python
np.linalg.svd(X_centered, full_matrices=False)
```

This gives the requested reduced/compact shapes.

------------------------------------------------------------------------

### Q6. How do you extract the first right singular vector?

**Answer:** Since NumPy returns `Vt = V.T`, the first right singular
vector can be obtained as:

``` python
first_right_singular_vector = Vt[0, :].flatten()
```

or, after constructing `V`:

``` python
V = Vt.T
first_right_singular_vector = V[:, 0]
```

The result should be a 1D array of shape:

``` text
(n_dimensions,)
```

------------------------------------------------------------------------

### Q7. How do you extract the first left singular vector?

**Answer:**

``` python
first_left_singular_vector = U[:, 0].flatten()
```

The result should be a 1D array of shape:

``` text
(n_samples,)
```

------------------------------------------------------------------------

## 2. Matrix rank and rank-one matrices

### Q8. What is matrix rank?

**Answer:** Matrix rank is the number of linearly independent rows or
columns in the matrix.

------------------------------------------------------------------------

### Q9. How do you create a rank-one matrix?

**Answer:** Given

$$
u \in \mathbb{R}^{5}, \qquad v \in \mathbb{R}^{4},
$$

construct

$$
A = uv^T.
$$

In NumPy:

``` python
u = np.array([1, 2, 3, 4, 5])
v = np.array([2, -1, 0, 3])

A = np.outer(u, v)
```

------------------------------------------------------------------------

### Q10. Why does $A=uv^T$ have rank 1?

**Answer:** Every row is a scalar multiple of the first row, and every
column is a scalar multiple of `u`. Therefore there is only one linearly
independent row/column, so the matrix has rank 1.

The supplied example gives:

``` text
A.shape = (5, 4)
rank = 1
```

------------------------------------------------------------------------

## 3. SVD reconstruction

### Q11. How do you reconstruct a matrix from its SVD?

**Answer:**

If

$$
A = UDV^T,
$$

then reconstruct with:

``` python
A_reconstructed = U @ D @ Vt
```

when `Vt` is the object returned by NumPy.

If you explicitly stored `V = Vt.T`, then:

``` python
A_reconstructed = U @ D @ V.T
```

------------------------------------------------------------------------

### Q12. What is the basic SVD reconstruction exercise from Day 7?

**Answer:** The task was:

1.  Create a random matrix `A` with shape `(50, 30)`.
2.  Compute SVD.
3.  Reconstruct `A` from the SVD.
4.  Create rank-1, rank-5, and rank-10 approximations.
5.  Calculate reconstruction error for each.
6.  Plot error as a function of rank.

The supplied hint was:

``` python
U, S, Vt = np.linalg.svd(A, full_matrices=False)
```

------------------------------------------------------------------------

### Q13. What are the shapes in the Day 7 SVD example?

**Answer:** For

``` python
A = rng.normal(size=(50, 30))
U, S, Vt = np.linalg.svd(A, full_matrices=False)
```

the supplied output was:

``` text
U.shape  = (50, 30)
S.shape  = (30,)
Vt.shape = (30, 30)
```

------------------------------------------------------------------------

## 4. Low-rank approximation

### Q14. What is a rank-$k$ approximation using SVD?

**Answer:** Keep only the first `k` singular components:

$$
A_k = U_{[:,1:k]}D_{1:k}V^T_{1:k,:}.
$$

In the zero-based Python indexing used in the files:

``` python
A_k = (
    U[:, :k]
    @ np.diag(S[:k])
    @ Vt[:k, :]
)
```

------------------------------------------------------------------------

### Q15. Why does a low-rank approximation lose information?

**Answer:** Low-rank approximation discards the smaller singular values
and their corresponding singular vectors. Since those components contain
part of the information in the original matrix, some information is
lost.

------------------------------------------------------------------------

### Q16. What happens to reconstruction error as rank increases?

**Answer:** The supplied Day 7 and Day 9 exercises show that the
reconstruction error decreases as more singular components are retained.

Therefore: - rank 1 has the largest approximation error among the tested
ranks, - larger ranks give smaller error, - the full rank reconstruction
gives the original matrix up to numerical precision.

------------------------------------------------------------------------

### Q17. In the Day 7 mini-exam, which approximation was best?

**Answer:** The supplied answer says the **rank-10** approximation was
best among rank 1, rank 3, and rank 10 because it had the lowest error,
reported as approximately:

``` text
36.72
```

Rank 1 was the worst of those choices.

------------------------------------------------------------------------

### Q18. What is the SVD compression problem from the Day 9 material?

**Answer:** The task was:

1.  Generate matrix `A` with shape `(80, 40)`.
2.  Compute SVD.
3.  Create rank-1, rank-5, rank-10, and rank-20 approximations.
4.  Calculate Frobenius reconstruction error.
5.  Plot error versus rank.

The supplied code pattern is:

``` python
A = rng.normal(size=(80, 40))

U, S, Vt = np.linalg.svd(A, full_matrices=False)

ranks = [1, 5, 10, 20]
errors = []

for k in ranks:
    A_k = (
        U[:, :k]
        @ np.diag(S[:k])
        @ Vt[:k, :]
    )
    error = np.linalg.norm(A - A_k, ord="fro")
    errors.append(error)
```

Then:

``` python
df_svd = pd.DataFrame({
    "rank": ranks,
    "error": errors
})
```

and plot:

``` python
plt.plot(df_svd["rank"], df_svd["error"], marker="o")
plt.xlabel("Rank")
plt.ylabel("Error")
plt.title("SVD approximation error")
plt.grid()
plt.show()
```

------------------------------------------------------------------------

### Q19. What is the standard error formula used for the SVD compression examples?

**Answer:** The Day 7 and Day 9 examples use the Frobenius norm:

``` python
error = np.linalg.norm(A - A_k, ord="fro")
```

For the January 2026 anomaly-detection exam problem, the reconstruction
error was instead calculated **row-wise using Euclidean distance**:

``` python
np.linalg.norm(X - X_approximation, axis=1)
```

------------------------------------------------------------------------

## 5. Explained variance from singular values

### Q20. How is explained variance defined from singular values?

**Answer:** For `N = n_dimensions`:

$$
\mathrm{EV}(k)
=
\frac{\sum_{i=1}^{k}\sigma_i^2}
{\sum_{i=1}^{N}\sigma_i^2}.
$$

The singular values are ordered from largest to smallest.

------------------------------------------------------------------------

### Q21. How do you compute cumulative explained variance in NumPy?

**Answer:**

``` python
d_squared = d**2

problem1_explained_variance = (
    np.cumsum(d_squared) / np.sum(d_squared)
)
```

This produces an increasing sequence whose final value is 1.

------------------------------------------------------------------------

### Q22. How do you find the smallest number of components explaining at least 99% variance?

**Answer:**

``` python
problem1_num_components = (
    np.argmax(problem1_explained_variance >= 0.99) + 1
)
```

The January 2026 exam data produced:

``` text
problem1_num_components = 10
```

The explained variance reached approximately:

``` text
k=1   0.16233339
k=2   0.31142268
k=3   0.42750646
k=4   0.53452990
k=5   0.63093263
k=6   0.72663179
k=7   0.80394305
k=8   0.87633669
k=9   0.94444296
k=10  0.99907483
```

So the first 10 components were the smallest number needed to reach at
least 99%.

------------------------------------------------------------------------

### Q23. How do you find the smallest number of components explaining at least 90% variance?

**Answer:** The Exam-2 notebook uses:

``` python
problem1_num_components = int(
    np.searchsorted(problem1_explained_variance, 0.90) + 1
)
```

The notebook does not supply the resulting numeric value in the relevant
solution cell, so the value should be computed from the actual digit
dataset rather than guessed.

------------------------------------------------------------------------

### Q24. What should the explained-variance curve look like for the January 2026 SVD exam dataset?

**Answer:** The supplied free-text answer says:

> The curve rises sharply, then quickly flattens. Most variance is
> captured by few components. The dataset has low effective
> dimensionality.

This matches the supplied values: the first 10 components already
explain about 99.91% of the variance.

------------------------------------------------------------------------

### Q25. How do you plot explained variance?

**Answer:**

``` python
y = problem1_explained_variance
x = np.arange(1, len(y) + 1)

plt.title("Explained variance")
plt.xlabel("Number of components")
plt.ylabel("EV(k)")
plt.plot(x, y)
plt.show()
```

------------------------------------------------------------------------

## 6. January 2026 exam: SVD + anomaly detection

### Q26. What was the main SVD problem in the January 16, 2026 exam?

**Answer:** The problem was about **SVD and anomaly detection using
low-rank reconstruction**.

The exam asked students to: 1. compute an SVD, 2. calculate explained
variance, 3. interpret the explained-variance curve, 4. construct a
low-rank approximation, 5. calculate reconstruction errors, 6. choose a
threshold producing exactly 100 outliers.

------------------------------------------------------------------------

### Q27. What data matrix was used in the January 2026 SVD exam problem?

**Answer:** The data were loaded from:

``` python
problem1_data = np.loadtxt(
    'data/SVD.csv',
    delimiter=','
)
```

The matrix has shape:

``` text
n_samples × n_dimensions
```

The supplied output shows:

``` text
n_samples = 1010
n_dimensions = 100
```

------------------------------------------------------------------------

### Q28. How was the January 2026 SVD computed?

**Answer:**

``` python
U, d, Vt = np.linalg.svd(problem1_data)

problem1_U = U
problem1_D = np.diag(d)
problem1_V = Vt.T
```

The first singular vectors were extracted with:

``` python
problem1_first_right_singular_vector = problem1_V[:, 0]
problem1_first_left_singular_vector = problem1_U[:, 0]
```

------------------------------------------------------------------------

### Q29. What singular values were observed in the January 2026 exam data?

**Answer:** The supplied output begins with:

``` text
398.607695
382.001409
337.075771
323.654157
307.175389
306.052410
275.082601
266.190143
258.187543
231.240973
```

Then the singular values drop dramatically to values around 12, 11, 10,
etc., followed by values close to numerical zero.

This large drop explains why only a small number of components are
needed to capture almost all the variance.

------------------------------------------------------------------------

### Q30. How do you construct the best rank-$k$ approximation in the exam?

**Answer:**

``` python
k = problem1_num_components

U_k = problem1_U[:, :k]
Vt_k = problem1_V[:k, :]
d_k = d[:k]
D_k = np.diag(d_k)

problem1_approximation = U_k @ D_k @ Vt_k
```

The resulting approximation has the same shape as the original data
matrix:

``` text
n_samples × n_dimensions
```

------------------------------------------------------------------------

### Q31. What is the row-wise reconstruction error in the exam?

**Answer:** For each row $i$:

$$
\|X_i-\hat X_i\|_2.
$$

The supplied code is:

``` python
problem1_reconstruction_error = np.linalg.norm(
    problem1_data - problem1_approximation,
    axis=1
)
```

The result has shape:

``` text
(n_samples,)
```

------------------------------------------------------------------------

### Q32. Why can reconstruction error be used for anomaly detection here?

**Answer:** The low-rank approximation represents the dominant structure
of the data. Rows that are poorly represented by this low-rank structure
have larger reconstruction errors. The exam therefore uses
reconstruction error to identify potential outliers.

------------------------------------------------------------------------

### Q33. How was the empirical distribution function used?

**Answer:** The exam asked students to plot the empirical distribution
function (EDF) of the reconstruction errors. The supplied instructions
say that `makeEDF` / `plotEDF` from `Utils.py` could be used.

------------------------------------------------------------------------

### Q34. How was the outlier threshold selected?

**Answer:** The threshold had to flag **exactly 100 samples**, using the
rule:

$$
\text{outlier if reconstruction error} \geq \text{threshold}.
$$

The supplied solution sorted the errors:

``` python
errors_asc_order = np.sort(problem1_reconstruction_error)
problem1_threshold = errors_asc_order[-100]
```

Then selected:

``` python
problem1_outliers = problem1_data[
    problem1_reconstruction_error >= problem1_threshold
]
```

and limited the result to 100 rows:

``` python
problem1_outliers = problem1_outliers[:100]
```

------------------------------------------------------------------------

### Q35. What threshold was obtained in the January 2026 exam solution?

**Answer:**

``` text
56.39649034649454
```

The supplied output also shows the largest reconstruction errors
reaching approximately:

``` text
72.9811
74.7597
83.7591
```

------------------------------------------------------------------------

## 7. PCA/SVD exam problem with handwritten digits

### Q36. What was the SVD/PCA task in the Exam-2 notebook?

**Answer:** The problem was about **PCA/SVD for handwritten digit
data**.

The data file was:

``` text
data/digits.csv
```

The first 64 columns are pixel intensities for an 8×8 handwritten digit
image, and the last column is the digit label.

The task included: 1. loading the data, 2. centering the feature matrix
column-wise, 3. computing compact SVD, 4. calculating cumulative
explained variance, 5. finding the smallest number of components
explaining at least 90%, 6. projecting onto the first two principal
directions, 7. interpreting the 2D PCA plot, 8. using the first `k` PCA
coordinates for nearest-centroid classification.

------------------------------------------------------------------------

### Q37. How is the digit feature matrix centered before SVD?

**Answer:**

``` python
problem1_X = data.iloc[:, :64].to_numpy(dtype=float)
problem1_y = data.iloc[:, 64].to_numpy()

problem1_X_centered = (
    problem1_X - problem1_X.mean(axis=0)
)
```

This subtracts the mean of each feature column separately.

------------------------------------------------------------------------

### Q38. How is the compact SVD of the centered digit matrix computed?

**Answer:**

``` python
U, singular_values, Vt = np.linalg.svd(
    problem1_X_centered,
    full_matrices=False
)

problem1_U = U
problem1_D = np.diag(singular_values)
problem1_V = Vt.T
```

Thus:

$$
X_c = UDV^T.
$$

------------------------------------------------------------------------

### Q39. How is explained variance computed in the digit PCA/SVD problem?

**Answer:**

``` python
singular_values = np.diag(problem1_D)

problem1_explained_variance = (
    np.cumsum(singular_values ** 2)
    / np.sum(singular_values ** 2)
)
```

The smallest number of components explaining at least 90% is:

``` python
problem1_num_components = int(
    np.searchsorted(
        problem1_explained_variance,
        0.90
    ) + 1
)
```

------------------------------------------------------------------------

### Q40. How are the first two PCA coordinates calculated?

**Answer:**

``` python
problem1_scores_2d = (
    problem1_X_centered @ problem1_V[:, :2]
)
```

The result has two columns, corresponding to the first two principal
directions.

------------------------------------------------------------------------

### Q41. What did the supplied answer say about the 2D PCA digit plot?

**Answer:** The supplied answer says that several digit classes should
form partially separated clusters, while some classes overlap.

PCA chooses directions that preserve as much **overall variance** as
possible; it does not optimize separation between digit labels.
Therefore, the first two principal components can reveal useful
structure, but they do not necessarily separate all ten digits
perfectly.

------------------------------------------------------------------------

### Q42. How are the first `k` PCA coordinates calculated for the nearest-centroid task?

**Answer:**

``` python
problem1_scores_k = (
    problem1_X_centered
    @ problem1_V[:, :problem1_num_components]
)
```

------------------------------------------------------------------------

### Q43. How are the digit centroids calculated in PCA space?

**Answer:** The first 80% of rows are used as training data and the
remaining 20% as test data.

For each digit `0,...,9`, the centroid is the mean of the training PCA
coordinates for that digit:

``` python
problem1_centroids = np.zeros(
    (10, problem1_num_components),
    dtype=float
)

for digit in range(10):
    problem1_centroids[digit] = (
        X_scores_train[y_train == digit].mean(axis=0)
    )
```

------------------------------------------------------------------------

### Q44. How are test points classified by nearest centroid?

**Answer:** Compute Euclidean distances from each test point to each
digit centroid:

``` python
distances = np.linalg.norm(
    X_scores_test[:, None, :]
    - problem1_centroids[None, :, :],
    axis=2
)
```

Then choose the nearest centroid:

``` python
problem1_test_predictions = np.argmin(
    distances,
    axis=1
)
```

The test accuracy is:

``` python
problem1_test_accuracy = float(
    np.mean(problem1_test_predictions == y_test)
)
```

------------------------------------------------------------------------

## 8. Complete exam coding templates

### Q45. What is the basic SVD template to memorize?

**Answer:**

``` python
import numpy as np

A = rng.normal(size=(50, 30))

U, S, Vt = np.linalg.svd(
    A,
    full_matrices=False
)

D = np.diag(S)
V = Vt.T

A_reconstructed = U @ D @ V.T
```

------------------------------------------------------------------------

### Q46. What is the basic rank-$k$ template to memorize?

**Answer:**

``` python
k = 5

A_k = (
    U[:, :k]
    @ np.diag(S[:k])
    @ Vt[:k, :]
)
```

------------------------------------------------------------------------

### Q47. What is the basic Frobenius reconstruction-error template?

**Answer:**

``` python
error = np.linalg.norm(
    A - A_k,
    ord="fro"
)
```

------------------------------------------------------------------------

### Q48. What is the basic row-wise reconstruction-error template used for anomaly detection?

**Answer:**

``` python
errors = np.linalg.norm(
    X - X_k,
    axis=1
)
```

------------------------------------------------------------------------

### Q49. What is the basic explained-variance template?

**Answer:**

``` python
S_squared = S ** 2

explained_variance = (
    np.cumsum(S_squared)
    / np.sum(S_squared)
)
```

------------------------------------------------------------------------

### Q50. How do you find the first component count reaching a variance threshold?

**Answer:**

For a 99% threshold:

``` python
k = np.argmax(
    explained_variance >= 0.99
) + 1
```

For the 90% threshold used in the digit notebook:

``` python
k = int(
    np.searchsorted(
        explained_variance,
        0.90
    ) + 1
)
```

------------------------------------------------------------------------

## 9. Short-answer revision questions

### Q51. What is the role of the singular values?

**Answer:** The singular values quantify the importance of the
corresponding singular components. Squared singular values are used in
the supplied explained-variance formula.

------------------------------------------------------------------------

### Q52. Why do the first few singular values matter most for low-rank approximation?

**Answer:** The supplied material uses the largest singular values
first. Keeping the first few components preserves the dominant
structure, while discarding smaller singular values loses less important
information.

------------------------------------------------------------------------

### Q53. What happens to approximation error when more singular values are retained?

**Answer:** The reconstruction error decreases as the rank increases in
the supplied exercises.

------------------------------------------------------------------------

### Q54. What does a rapidly rising explained-variance curve mean?

**Answer:** It means that a small number of components capture most of
the variance. In the January 2026 dataset, the curve rises sharply and
then flattens, indicating low effective dimensionality.

------------------------------------------------------------------------

### Q55. What does a flat explained-variance curve mean?

**Answer:** It means variance is spread more gradually over the
components, so more components are needed to capture a specified
fraction of the variance.

------------------------------------------------------------------------

### Q56. What is the relationship between SVD and low-rank approximation?

**Answer:** SVD provides the singular components that are retained or
discarded to form a low-rank approximation. Keeping only the first `k`
components produces the rank-`k` approximation used throughout the
supplied material.

------------------------------------------------------------------------

### Q57. What is the relationship between SVD and PCA in the supplied digit problem?

**Answer:** The digit problem centers the feature matrix first and then
uses the right singular vectors from the SVD as the principal
directions. The PCA coordinates are obtained by projecting the centered
data onto those directions:

``` python
problem1_X_centered @ problem1_V[:, :k]
```

------------------------------------------------------------------------

### Q58. Does PCA/SVD automatically separate classes?

**Answer:** No. The supplied digit answer explicitly states that PCA
preserves overall variance rather than optimizing separation between
digit labels. Therefore, PCA can reveal structure and partially
separated clusters without perfectly separating all classes.

------------------------------------------------------------------------

## 10. Exam-style questions

### Q59. Given `X = UDVᵀ`, what does `Vt` represent in NumPy?

**Answer:** `Vt` represents $V^T$, not $V$. Therefore:

``` python
V = Vt.T
```

------------------------------------------------------------------------

### Q60. If `S = [10, 5, 0]`, what is the cumulative explained variance after the first component?

**Answer:**

$$
\frac{10^2}{10^2+5^2+0^2}
=
\frac{100}{125}
=
0.8.
$$

So the first component explains 80% of the variance.

------------------------------------------------------------------------

### Q61. With `S = [10, 5, 0]`, what is the cumulative explained variance after two components?

**Answer:**

$$
\frac{10^2+5^2}{10^2+5^2+0^2}
=
1.
$$

So two components explain 100% of the variance.

------------------------------------------------------------------------

### Q62. If a rank-1 approximation has larger reconstruction error than a rank-10 approximation, which one is better for reconstruction?

**Answer:** The rank-10 approximation is better because it has the lower
reconstruction error.

------------------------------------------------------------------------

### Q63. If exactly 100 outliers are required and the rule is `error >= threshold`, how can you choose the threshold?

**Answer:** Sort the reconstruction errors and use the 100th largest
error:

``` python
threshold = np.sort(errors)[-100]
```

Then select:

``` python
outliers = X[errors >= threshold]
```

The January 2026 supplied solution used exactly this approach.

------------------------------------------------------------------------

### Q64. What should you check if your SVD matrix multiplication gives a shape error?

**Answer:** Check the shapes of: - `U` - `S` - `D = np.diag(S)` - `Vt` -
`V = Vt.T`

The supplied course hints repeatedly recommend using:

``` python
print(U.shape)
print(S.shape)
print(Vt.shape)
```

and checking shapes before continuing.

------------------------------------------------------------------------

## 11. Final SVD memory sheet

### Core decomposition

$$
X = UDV^T
$$

### NumPy

``` python
U, S, Vt = np.linalg.svd(
    X,
    full_matrices=False
)
```

### Diagonal matrix

``` python
D = np.diag(S)
```

### Right singular vectors

``` python
V = Vt.T
```

### First right singular vector

``` python
V[:, 0]
```

### First left singular vector

``` python
U[:, 0]
```

### Explained variance

$$
EV(k)
=
\frac{\sum_{i=1}^{k}\sigma_i^2}
{\sum_{i=1}^{N}\sigma_i^2}
$$

``` python
ev = np.cumsum(S**2) / np.sum(S**2)
```

### Rank-$k$ approximation

``` python
A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

### Frobenius reconstruction error

``` python
np.linalg.norm(A - A_k, ord="fro")
```

### Row-wise reconstruction error

``` python
np.linalg.norm(X - X_k, axis=1)
```

### Outlier threshold for exactly 100 samples

``` python
threshold = np.sort(errors)[-100]
```

### Main interpretation

-   Large singular values = dominant components.
-   Keeping the first few components gives a low-rank approximation.
-   Discarding smaller singular values loses information.
-   More retained components reduce reconstruction error.
-   A sharply rising explained-variance curve means most variance is
    captured by few components.
-   SVD/PCA can reduce dimensionality but does not necessarily produce
    perfect class separation.
-   In anomaly detection using low-rank reconstruction, unusually large
    reconstruction error can be used to flag outliers.

## 12. Supplied numerical results worth memorizing

### January 2026 SVD exam

-   Data shape: `1010 × 100`
-   Number of components needed for at least 99% explained variance:
    **10**
-   EV after 1 component: approximately **0.16233339**
-   EV after 9 components: approximately **0.94444296**
-   EV after 10 components: approximately **0.99907483**
-   Outlier count required: **100**
-   Reconstruction-error threshold: **56.39649034649454**

### Day 7 rank-approximation example

-   Rank-10 was the best among rank 1, 3, and 10 in the supplied
    example.
-   Reported rank-10 error: approximately **36.72**.

## 13. Source scope

This study file is based on the uploaded: -
`MK_1MS041DataScience_preparation_REFERENCE_MERGED_UPDATED.md` -
`MK_1MS041DataScience_day10_exam20260116.md` -
`Exam-2_solved_autograde_friendly.ipynb`

The SVD-specific material found in the preparation compilation is
concentrated in the Day 7 SVD/geometry material, the Day 9 SVD
compression material, the January 16, 2026 SVD exam problem, and the
Exam-2 PCA/SVD handwritten-digit problem. The first preparation file
contains the broader course preparation material but does not add a
separate SVD section beyond what is represented in the compiled SVD
material.
