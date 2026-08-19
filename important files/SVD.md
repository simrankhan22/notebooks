
This problem is about **SVD** and a simple **anomaly detection** idea using low-rank reconstruction.



Unless stated otherwise, when you are asked to produce a matrix or vector, it must be a **NumPy array**.

1. **[4p] SVD.** Load `data/SVD.csv` as instructed in the code cell. Let $X$ be the data matrix of shape `n_samples × n_dimensions`. Compute an SVD
   $$X = U D V^T$$
   where $U$ has shape `n_samples × n_dimensions`, $D$ is the diagonal matrix of shape `(n_dimensions,n_dimensions)` that has the singular values on the diagonal (see documentation for `np.diag`), and $V$ has shape `n_dimensions × n_dimensions`.
   **Important:** `np.linalg.svd` returns `U, d, Vt` where `Vt` is $V^T$.
   Also extract the **first** right and left singular vectors and store them as 1D arrays in the variables provided.

2. **[3p] Explained variance.** For $N =$ `n_dimensions`, define the explained variance using the first $k$ singular values as
   $$
   \mathrm{EV}(k) = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^N \sigma_i^2}.
   $$
   Compute $\mathrm{EV}(k)$ for $k=1,2,\dots,N$ and store it in `problem1_explained_variance` (length `N`). Then set `problem1_num_components` to the **smallest** $k$ such that $\mathrm{EV}(k) \ge 0.99$.

3. **[3p] Plot + interpretation.** Plot explained variance (x-axis: number of components $k$, y-axis: $\mathrm{EV}(k)$). In the markdown cell below, reason about the shape of the curve for this dataset.

4. **[4p] Low-rank reconstruction + outliers.**
   - Using `problem1_num_components`, construct the best rank-$k$ approximation of $X$ and store it in `problem1_approximation`.
   - Compute the row-wise Euclidean reconstruction error $\|X_i - \hat X_i\|_2$ for each row $i$ and store it in `problem1_reconstruction_error` (shape `(n_samples,)`).
   - Plot the empirical distribution function (EDF) of the reconstruction errors (you may use `makeEDF` / `plotEDF` from `Utils.py`).
   - Choose a threshold `problem1_threshold` so that **exactly 100** samples are flagged as outliers (i.e. have reconstruction error >= the threshold).
   - Store those flagged rows in `problem1_outliers` (shape `(100, n_dimensions)`).


```
# Part 1: 4 points

# Load the data from the file data/SVD.csv and store the data in a numpy array called problem1_data below
# Double check that the numbers have been parsed correctly by checking the dtype of the array by calling problem1_data.dtype
#problem1_data = XXX # A numpy array of shape n_samples x n_dimensions

#problem1_U = XXX # The matrix of left singular vectors of problem1_data with shape n_samples x n_dimensions
#problem1_D = XXX # The diagonal matrix with the singular values of problem1_data on the diagonal with shape n_dimensions x n_dimensions
#problem1_V = XXX # The matrix of right singular vectors of problem1_data with shape n_dimensions x n_dimensions

#problem1_first_right_singular_vector = XXX # The first right singular vector of problem1_data with shape (n_dimensions,) hint sometimes one needs to invoke flatten() to avoid having shape (n_dimensions, 1) or (1, n_dimensions)

#problem1_first_left_singular_vector = XXX # The first left singular vector of problem1_data with shape (n_samples,) hint sometimes one needs to invoke flatten() to avoid having shape (n_samples, 1) or (1, n_samples)
import numpy as np
import matplotlib.pyplot as plt
#from np.linalg import svd

# Load data
# Replace with actual path
problem1_data = np.loadtxt('data/SVD.csv', delimiter=',')
#print(X)

# TODO: Compute SVD
#U, d, Vt = None, None, None
U, d, Vt = np.linalg.svd(problem1_data)
print("U=", U)
print("d=", d)
print("Vt=", Vt)
problem1_U = U
problem1_D = np.diag(d)
problem1_V = Vt.T

# TODO: Extract first singular vectors
problem1_first_right_singular_vector = problem1_V[:,0]
problem1_first_left_singular_vector = problem1_U[:,0]
print("V=", problem1_first_right_singular_vector)
print("D=", problem1_first_left_singular_vector)

# print(X.shape)
# print(U.shape)
# print(D.shape)
# print(V.shape)
# print(first_right.shape)
# print(first_left.shape)

# Part 2: 3 points

# Calculate the explained variance of using 1,2,3,...,n_dimensions singular values and store it as a numpy array called problem1_explained_variance below
#problem1_explained_variance = XXX # A numpy array of shape (n_dimensions,), it should be an increasing sequence of positive numbers and the last element should be 1

# Store in the variable below the smallest number of singular values needed to explain at least 99% of the variance
#problem1_num_components = XXX # An integer

# squared singular values
d_squared = d**2

# cumulative explained variance
problem1_explained_variance = np.cumsum(d_squared) / np.sum(d_squared)

# smallest k such that EV(k) >= 0.99
problem1_num_components = np.argmax(problem1_explained_variance >= 0.99) + 1
print(problem1_explained_variance)
print(problem1_num_components)
print(problem1_explained_variance.size)


# Part 3: 3 points

# Put the code below to plot the explained variance
# use for instance matplotlib
# XXX
# XXX
# XXX

# Plot explained variance
import numpy as np 
import matplotlib.pyplot as plt 

y = problem1_explained_variance
#x = np.arange(1,explained_variance.size+1) 
x = np.arange(1,len(y)+1) 

# plotting
plt.title("Explained variance") 
plt.xlabel("Number of components") 
plt.ylabel("EV(k)") 
plt.plot(x, y, color ="red") 
plt.show()



# Part 4: 4 points

# Calculate the approximating matrix of problem1_data using the first problem1_num_components singular values and store it in the variable below
#problem1_approximation = XXX # A numpy array of shape n_samples x n_dimensions

# Calculate the reconstruction error of problem1_data using problem1_approximation and store it in the variable below (should have shape (n_samples,)) (row wise Euclidean distance)
#problem1_reconstruction_error = XXX

# Put the code below to plot the empirical distribution function of the reconstruction error
# You can use the Utils.py file for plotting the empirical distribution function, makeEDF and plotEDF functions
# XXX
# XXX
# XXX


# Store the value of the selected threshold in the variable below
#problem1_threshold = XXX

# Finally store the samples of problem1_data that have a reconstruction error larger than problem1_threshold in the variable below, should have shape (100, n_dimensions)
#problem1_outliers = XXX



# Reconstruction and errors
# TODO
k=problem1_num_components
U_k=problem1_U[:,:k]     #first column
Vt_k=problem1_V[:k,:]   #first row
d_k=d[:k]
D_k=np.diag(d_k)
#D is diagonal matrix
print(U_k.shape)
print(D_k.shape)
print(Vt_k.shape)
problem1_approximation = U_k@D_k@Vt_k
print(problem1_approximation)
#errors = None
#errors = np.sqrt(np.sum((X - X_k)**2, axis=1))
problem1_reconstruction_error = np.linalg.norm(problem1_data - problem1_approximation, axis=1)
print("errors=", problem1_reconstruction_error)

# TODO: choose threshold for 100 outliers
errors_asc_order = np.sort(problem1_reconstruction_error)
print(errors_asc_order)
problem1_threshold = errors_asc_order[-100]
print("treshold=", problem1_threshold)
problem1_outliers = problem1_data[problem1_reconstruction_error >=problem1_threshold]
problem1_outliers= problem1_outliers[:100]
print("outliers=", problem1_outliers)


```
## Problem 4 — SVD Compression

Tasks:
1. Generate matrix `A` with shape `(80, 40)`.
2. Compute SVD.
3. Create rank-1, rank-5, rank-10, rank-20 approximations.
4. Calculate Frobenius reconstruction error.
5. Plot error vs rank.

Hints:
- `U, S, Vt = np.linalg.svd(A, full_matrices=False)`
- `A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]`

```
A = rng.normal(size=(80, 40))

U, S, Vt = np.linalg.svd(A, full_matrices=False)

ranks = [1, 5, 10, 20]
errors = []

for k in ranks:
    A_k = U[:, :k] @np.diag(S[:k]) @Vt[:k, :]
    error = np.linalg.norm(A -A_k, ord="fro")
    errors.append(error)

df_svd = pd.DataFrame({"rank": ranks, "error": errors})
df_svd


plt.plot(df_svd["rank"], df_svd["error"], marker="o")
plt.xlabel("Rank")
plt.ylabel("Error")
plt.title("SVD approximation error")
plt.grid()
plt.show()
```

## Problem 1 — Near Orthogonality in High Dimensions

Generate random vectors in dimensions `d = 10, 100, 1000`.

Tasks:
1. Generate `R = 5000` pairs of random normal vectors for each dimension.
2. Normalize vectors.
3. Compute dot product for each pair.
4. Store mean and standard deviation of dot products.
5. Plot histogram for each dimension.
6. Explain what happens when dimension grows.

Hints:
- Dot product of normalized vectors is cosine similarity.
- Normalize by dividing by vector norm.
- `np.linalg.norm(x, axis=1)` gives row norms.
- For row-wise dot product use `np.sum(x * y, axis=1)`.

```
d_values = [10, 100, 1000]
R = 5000

results = []

for d in d_values:
    X = rng.normal(size=(R, d))
    Y = rng.normal(size=(R, d))

    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)

    #row-wise dot product
    dots = np.sum(X_norm * Y_norm, axis=1)

    results.append({
        "d": d,
        "mean_dot": np.mean(dots),
        "std_dot": np.std(dots)
    })

    plt.figure()
    plt.hist(dots, bins=40)
    plt.xlabel("Dot product")
    plt.ylabel("Frequency")
    plt.title(f"Dot products for d={d}")
    plt.grid()
    plt.show()

pd.DataFrame(results)

```
## Problem 3 — SVD and Low-Rank Approximation

Tasks:
1. Create a random matrix `A` with shape `(50, 30)`.
2. Compute SVD.
3. Reconstruct `A` from SVD.
4. Create rank-1, rank-5, rank-10 approximations.
5. Calculate reconstruction error for each approximation.
6. Plot error as function of rank.

Hints:
- Use `U, S, Vt = np.linalg.svd(A, full_matrices=False)`.
- Rank-k approximation:
  `A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]`
- Error:
  `np.linalg.norm(A - A_k, ord="fro")`

```
A = rng.normal(size=(50, 30))

U, S, Vt = np.linalg.svd(A,full_matrices=False)

# Check shapes
print(U.shape, S.shape, Vt.shape)

ranks = [1, 5, 10, 20, 30]
errors = []

for k in ranks:
    A_k = U[:, :k] @np.diag(S[:k]) @Vt[:k, :]
    error = np.linalg.norm(A -A_k, ord="fro")
    errors.append(error)

df_errors = pd.DataFrame({
    "rank": ranks,
    "error": errors
})

df_errors

plt.plot(df_errors["rank"], df_errors["error"], marker="o")
plt.xlabel("Rank")
plt.ylabel("Reconstruction error")
plt.title("Low-rank approximation error")
plt.grid()
plt.show()


```
