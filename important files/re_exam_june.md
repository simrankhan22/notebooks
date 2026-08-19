# Exam 16th of June 2026, 8.00-13.00 for the course 1MS041 (Introduction to Data Science / Introduktion till dataanalys)

## Instructions:
1. Complete the problems by following instructions.
2. When done, submit this file with your solutions saved, following the instruction sheet.

This exam has 3 problems for a total of 40 points, to pass you need
20 points. The bonus will be added to the score of the exam and rounded afterwards.

## Some general hints and information:
* Try to answer all questions even if you are uncertain.
* Comment your code, so that if you get the wrong answer I can understand how you thought
this can give you some points even though the code does not run.
* Follow the instruction sheet rigorously.
* This exam is partially autograded, but your code and your free text answers are manually graded anonymously.
* If there are any questions, please ask the exam guards, they will escalate it to me if necessary.

## Tips for free text answers
* Be VERY clear with your reasoning, there should be zero ambiguity in what you are referring to.
* If you want to include math, you can write LaTeX in the Markdown cells, for instance `$f(x)=x^2$` will be rendered as $f(x)=x^2$ and `$$f(x) = x^2$$` will become an equation line, as follows
$$f(x) = x^2$$
Another example is `$$f_{Y \mid X}(y,x) = P(Y = y \mid X = x) = \exp(\alpha \cdot x + \beta)$$` which renders as
$$f_{Y \mid X}(y,x) = P(Y = y \mid X = x) = \exp(\alpha \cdot x + \beta)$$

## Finally some rules:
* You may not communicate with others during the exam, for example:
    * You cannot ask for help in Stack-Overflow or other such help forums during the Exam.
    * You may not communicate with AIs, for instance ChatGPT.
    * Your on-line and off-line activity is being monitored according to the examination rules.

## Good luck!

# Insert your anonymous exam ID as a string in the variable below
examID="XXX"

---
## Exam vB, PROBLEM 1
Maximum Points = 14

This problem is about **PCA/SVD** for handwritten digit data. Unless stated otherwise, every vector or matrix you create should be a NumPy array.

The file `data/digits.csv` contains one row per image. The first 64 columns are pixel intensities for an 8 by 8 handwritten digit image, and the last column `target` is the digit label.

1. **[4p] Data and SVD.** Load the data. Store the feature matrix in `problem1_X` and the labels in `problem1_y`. Center the feature matrix column-wise and store it in `problem1_X_centered`. Compute the compact SVD
   $$X_c = U D V^T$$
   and store the matrices in `problem1_U`, `problem1_D`, and `problem1_V`, where `problem1_D` is a square diagonal matrix and `problem1_V` contains the right singular vectors as columns. If `problem1_X` has shape `(n_samples, 64)`, the compact shapes should be `problem1_U.shape == (n_samples, 64)`, `problem1_D.shape == (64, 64)`, and `problem1_V.shape == (64, 64)`. If you use `np.linalg.svd`, remember that compact SVD means `full_matrices=False`.

2. **[3p] Explained variance.** Compute the cumulative explained variance from the singular values on the diagonal of `problem1_D`, ordered from largest to smallest, and store it in `problem1_explained_variance`. If the singular values are $d_1 \geq d_2 \geq \cdots \geq d_{64}$, then the cumulative explained variance after the first $k$ components is
   $$
   \frac{\sum_{j=1}^k d_j^2}{\sum_{j=1}^{64} d_j^2}.
   $$
   Thus `problem1_explained_variance[k-1]` should contain this value. Store in `problem1_num_components` the smallest number of components needed to explain at least 90% of the variance.

3. **[3p] Two-dimensional PCA coordinates and interpretation.** Store the projection onto the first two principal directions in `problem1_scores_2d`. Plot these coordinates and color the points by the digit labels. In the markdown cell below, briefly explain what the plot suggests and why PCA can or cannot separate all digits perfectly.

4. **[4p] Nearest-centroid classification in PCA space.** Use the centered data and PCA directions already computed above; do not recompute PCA after the train/test split. Store the first `problem1_num_components` PCA coordinates in `problem1_scores_k`. Use a deterministic 80/20 split where the first 80% of the rows are training rows and the remaining 20% are test rows. For each digit label `0, 1, ..., 9`, compute the centroid of the training points in PCA space and store these ten centroids in `problem1_centroids`, with row `i` corresponding to digit `i`. Classify each test row by the nearest centroid in Euclidean distance and store the predicted labels in `problem1_test_predictions`, in increasing row-index order. Store the test accuracy in `problem1_test_accuracy`.

```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
X = data.drop(columns=['target'], axis=1)
print(X)


y = data.target
print(y)
center_function = lambda x: x - meanVal

# finding the mean value of the data
meanVal = X.mean()

centerData = center_function(data)
print(f"The array of centered data values is \n{centerData}")

print('data shape:', centerData.shape)
X_new = centerData.drop(columns=['target'], axis=1)
print(X_new)

import numpy as np
import pandas as pd

def load_dataset(file_path=None):
    """
    Loads a dataset from a CSV file. Returns a numeric NumPy array.
    """
    if file_path:
        df = pd.read_csv('data/digits.csv', encoding='latin1')
    else:
        import seaborn as sns
        df = sns.load_dataset('iris').drop(columns=['species'])
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.to_numpy()

def compute_svd(matrix):
    """Computes full SVD: returns U, d (1D singular values), Vt."""
    U, d, Vt = np.linalg.svd(matrix, full_matrices=True)
    return U, d, Vt
# Part 1: 4 points

# Load data
problem1_data = load_dataset('data/digits.csv')  # shape: n_samples x n_dimensions
print('dtype:', problem1_data.dtype, '| shape:', problem1_data.shape)

n_samples, n_dimensions = problem1_data.shape

# Compute SVD: X = U D V^T
# np.linalg.svd returns U (n_samples x n_dimensions), d (n_dimensions,), Vt (n_dimensions x n_dimensions)
U_full, d_vals, Vt_full = np.linalg.svd(problem1_data, full_matrices=False)

problem1_U1 = U_full                          # shape: n_samples x n_dimensions
problem1_D1 = np.diag(d_vals)                 # shape: n_dimensions x n_dimensions
problem1_V1 = Vt_full.T                       # shape: n_dimensions x n_dimensions (V, not V^T)

# First right singular vector = first row of Vt, shape (n_dimensions,)
problem1_first_right_singular_vector = Vt_full[0, :].flatten()

# First left singular vector = first column of U, shape (n_samples,)
problem1_first_left_singular_vector = U_full[:, 0].flatten()

print('U shape:', problem1_U1.shape)
print('D shape:', problem1_D1.shape)
print('V shape:', problem1_V1.shape)
print('First right singular vector shape:', problem1_first_right_singular_vector.shape)
print('First left singular vector shape:', problem1_first_left_singular_vector.shape)
sigma_sq = d_vals ** 2
cumulative_ev = np.cumsum(sigma_sq) / np.sum(sigma_sq)

# EV(k) for k = 1, 2, ..., N  (stored as length-N array)
problem1_explained_variance1 = cumulative_ev  # increasing sequence ending at 1.0

# Smallest k such that EV(k) >= 0.90
problem1_num_components1 = int(np.searchsorted(cumulative_ev, 0.90) + 1)

print('Explained variance (first 10):', problem1_explained_variance[:10])
print('Num components for 90% variance:', problem1_num_components)
# Part 1: Data, column-wise centering, and compact SVD

data = pd.read_csv("data/digits.csv", encoding="latin1")

# First 64 columns are pixel features; last column is the digit label.
problem1_X = data.iloc[:, :64].to_numpy(dtype=float)
problem1_y = data.iloc[:, 64].to_numpy()

# Center each feature column separately.
problem1_X_centered = problem1_X - problem1_X.mean(axis=0)

# Compact SVD: X_c = U D V^T
U, singular_values, Vt = np.linalg.svd(
    problem1_X_centered,
    full_matrices=False
)

problem1_U = U
problem1_D = np.diag(singular_values)
problem1_V = Vt.T
# Part 2: Explained variance

singular_values = np.diag(problem1_D)

problem1_explained_variance = (
    np.cumsum(singular_values ** 2) /
    np.sum(singular_values ** 2)
)

problem1_num_components = int(
    np.searchsorted(problem1_explained_variance, 0.90) + 1
)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_numeric = pd.factorize(y)[0]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_numeric, cmap='coolwarm', edgecolor='k', s=80)
plt.xlabel('Original Feature 1')
plt.ylabel('Original Feature 2')
plt.title('Before PCA: Using First 2 Standardized Features')
plt.colorbar(label='Target classes')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_numeric, cmap='coolwarm', edgecolor='k', s=80)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('After PCA: Projected onto 2 Principal Components')
plt.colorbar(label='Target classes')

plt.tight_layout()
plt.show()
# Part 3: Two-dimensional PCA coordinates and plot

problem1_scores_2d = (
    problem1_X_centered @ problem1_V[:, :2]
)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    problem1_scores_2d[:, 0],
    problem1_scores_2d[:, 1],
    c=problem1_y,
    cmap="tab10",
    s=12,
    alpha=0.7
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Digits projected onto the first two principal components")
plt.colorbar(scatter, label="Digit")
plt.show()
# Importing the turicreate Library
X_train, X_test, y_train, y_test = train_test_split(centerData, y, test_size=0.2, random_state=42)

# Print the number of samples in the training and testing sets
print(f"Number of samples in the training set: {len(X_train)}")
print(f"Number of samples in the testing set: {len(X_test)}")
## Free text answer for Part 3

The 2D PCA plot should show that several digit classes form partially separated clusters, while some classes overlap. PCA chooses directions that preserve as much overall variance as possible; it does not optimize separation between digit labels. Therefore, the first two principal components can reveal useful structure, but they do not necessarily separate all ten digits perfectly.
# Part 4: k-dimensional PCA and nearest-centroid classification

# Use the PCA directions already computed from the full centered dataset.
problem1_scores_k = (
    problem1_X_centered @ problem1_V[:, :problem1_num_components]
)

n_samples = problem1_X.shape[0]
split_index = int(0.8 * n_samples)

# First 80% of rows = training; final 20% = test.
X_scores_train = problem1_scores_k[:split_index]
X_scores_test = problem1_scores_k[split_index:]
y_train = problem1_y[:split_index]
y_test = problem1_y[split_index:]

# One centroid for each digit 0,...,9.
problem1_centroids = np.zeros(
    (10, problem1_num_components),
    dtype=float
)

for digit in range(10):
    problem1_centroids[digit] = X_scores_train[y_train == digit].mean(axis=0)

# Euclidean distance from every test point to every centroid.
distances = np.linalg.norm(
    X_scores_test[:, None, :] -
    problem1_centroids[None, :, :],
    axis=2
)

problem1_test_predictions = np.argmin(distances, axis=1)
problem1_test_accuracy = float(
    np.mean(problem1_test_predictions == y_test)
)
---
#### Local Test for Exam vB, PROBLEM 1
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.

# Optional local format checks for Problem 1. These checks do not prove correctness,
# but they are meant to catch the most common SVD shape issue.
import numpy as np

try:
    assert isinstance(problem1_X, np.ndarray)
    assert problem1_X.shape[1] == 64
    assert isinstance(problem1_y, np.ndarray)
    assert problem1_y.shape[0] == problem1_X.shape[0]

    n_samples, n_dimensions = problem1_X.shape
    expected_U_shape = (n_samples, n_dimensions)
    expected_D_shape = (n_dimensions, n_dimensions)
    expected_V_shape = (n_dimensions, n_dimensions)

    print("Expected compact SVD shapes:")
    print("  problem1_U:", expected_U_shape)
    print("  problem1_D:", expected_D_shape)
    print("  problem1_V:", expected_V_shape)
    print("Your shapes:")
    print("  problem1_U:", getattr(problem1_U, 'shape', None))
    print("  problem1_D:", getattr(problem1_D, 'shape', None))
    print("  problem1_V:", getattr(problem1_V, 'shape', None))

    if getattr(problem1_U, 'shape', None) == (n_samples, n_samples):
        print("Warning: problem1_U has the full SVD shape. NumPy uses full_matrices=True by default.")
        print("Use np.linalg.svd(problem1_X_centered, full_matrices=False) for the compact SVD.")

    assert problem1_U.shape == expected_U_shape
    assert problem1_D.shape == expected_D_shape
    assert problem1_V.shape == expected_V_shape
    assert np.allclose(problem1_X_centered, problem1_U @ problem1_D @ problem1_V.T, atol=5e-3)
    assert problem1_scores_2d.shape == (problem1_X.shape[0], 2)

    k = int(problem1_num_components)
    split_index = int(0.8 * n_samples)
    n_test = n_samples - split_index
    assert problem1_scores_k.shape == (n_samples, k)
    assert problem1_centroids.shape == (10, k)
    assert problem1_test_predictions.shape == (n_test,)
    assert 0 <= float(problem1_test_accuracy) <= 1
    print("Problem 1 format checks passed.")
except Exception as error:
    print("Problem 1 format check failed:", error)

---
```

## Exam vB, PROBLEM 2
Maximum Points = 12

This problem is about **linear regression** and evaluating prediction error on held-out data. Unless stated otherwise, every vector or matrix you create should be a NumPy array.

The file `data/auto.csv` contains car measurements. We will predict fuel efficiency `mpg` from the features `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, and `model-year`, in that order.

1. **[2p] Load and clean data.** Load the file, remove rows where `horsepower` is missing, store the feature matrix in `problem2_X` and the target vector in `problem2_y`. Missing horsepower values are encoded as `?` or as an empty value.

2. **[2p] Train/test split and standardization.** Let `problem2_split_index = int(0.8 * n)`, where `n` is the number of cleaned rows. Use rows before this index as the training set and the remaining rows as the test set. Store the four arrays `problem2_X_train`, `problem2_X_test`, `problem2_y_train`, and `problem2_y_test`. Standardize features using the training mean and training standard deviation only, computed with NumPy's default `np.std(..., axis=0)`. Store the standardized train and test matrices in `problem2_X_train_standardized` and `problem2_X_test_standardized`.

3. **[3p] Fit linear regression.** Fit linear regression with an intercept using least squares on the standardized training features. Store the coefficient vector, including the intercept as the first entry, in `problem2_beta`. If you use `sklearn.linear_model.LinearRegression`, the intercept is stored in `model.intercept_` and the feature coefficients are stored in `model.coef_`, so the required order is `[model.intercept_, model.coef_[0], ..., model.coef_[5]]`. Store test predictions in `problem2_y_pred_test` and residuals `y_test - y_pred` in `problem2_residuals_test`.

4. **[3p] Test metrics and baseline.** Compute test MSE, MAE, and R^2 in `problem2_mse_test`, `problem2_mae_test`, and `problem2_r2_test`. Also compute `problem2_baseline_mse_test` for the predictor that always predicts the training-set mean of `mpg`, and set `problem2_model_beats_baseline` to `True` exactly when the linear model has smaller test MSE.

5. **[2p] Hoeffding interval.** Clip the absolute residuals

   at 50 and compute their empirical mean on the test set. Construct a two-sided Hoeffding confidence interval with confidence level 95% for the expected clipped absolute error. You may use `epsilon_bounded` from `Utils.py`, but you should decide its arguments from the sample size, the bound, and the confidence level. Since the clipped absolute error is between 0 and 50, intersect your final interval with `[0, 50]`. Store the interval endpoints in `problem2_lower_bound` and `problem2_upper_bound`.

```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
csv_path = "data/auto.csv"

# Read the CSV file
df = pd.read_csv(csv_path).dropna()
problem2_df = df

df.head()
clean_data = df.dropna()
print(clean_data)

problem2_df.columns = problem2_df.columns.str.replace('\ufeff', '', regex=False).str.strip()
#problem2_features = problem2_df.columns.tolist()[:-1]
problem2_features = clean_data[['cylinders', 'displacement', 'horsepower','weight','acceleration','model-year']]
print("Features:", problem2_features)
# Fill in the target as a string with the correct column name

#problem2_target = problem2_df.columns.tolist()[-1]
problem2_target = " mpg"
print("Target:", problem2_target)
# Part 1: Load and clean data

df2 = pd.read_csv("data/auto.csv")

# Missing horsepower is represented by '?' or an empty value.
df2 = df2.replace({"?": np.nan, "": np.nan})
df2["horsepower"] = pd.to_numeric(
    df2["horsepower"], errors="coerce"
)
df2 = df2.dropna(subset=["horsepower"]).copy()

feature_columns = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model-year"
]

problem2_X = df2[feature_columns].to_numpy(dtype=float)
problem2_y = df2["mpg"].to_numpy(dtype=float)
import numpy as np
from sklearn.model_selection import train_test_split
X = problem2_df[['cylinders', 'displacement', 'horsepower','weight','acceleration','model-year']].to_numpy(dtype='float64')  
y = problem2_df['mpg'].to_numpy(dtype='int64')

#make split, keep size as 0.8
problem2_X_train,problem2_X_test,problem2_y_train,problem2_y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)

#display
print("Columns:",problem2_df.columns.tolist())
print("X", X)
print("Y:", y)
new_df = clean_data # in this new dataset we only take 'Volkswagen' Cars
print(new_df.shape) # Viewing the new dataset shape
print(new_df.isnull().sum()) # Is there any Null or Empty cell presents
new_df = new_df.dropna() # Deleting the rows which have Empty cells
print(new_df.shape) # After deletion Vewing the shape
print(new_df.isnull().sum()) #Is there any Null or Empty cell presents
new_df.sample(2) # Checking the random dataset sample

# Define features and target variable
X = df.drop('mpg', axis=1)
y = df['mpg']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Create the model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Display the coefficients
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_std_train = scaler.transform(X)

scaler = StandardScaler().fit(X_test)
X_std_test = scaler.transform(X)
# Make predictions
y_pred = model.predict(X_test)
print(y_pred)
# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R-squared:', r2)
baseline_mse = mean_squared_error(y_test, y_pred)

residuals_test = y_test - y_pred
print(residuals_test)
# Part 2: Contiguous 80/20 split and training-only standardization

n = problem2_X.shape[0]
problem2_split_index = int(0.8 * n)

problem2_X_train = problem2_X[:problem2_split_index]
problem2_X_test = problem2_X[problem2_split_index:]

problem2_y_train = problem2_y[:problem2_split_index]
problem2_y_test = problem2_y[problem2_split_index:]

# Compute mean and standard deviation from TRAINING data only.
problem2_train_mean = problem2_X_train.mean(axis=0)
problem2_train_std = problem2_X_train.std(axis=0)

problem2_X_train_standardized = (
    problem2_X_train - problem2_train_mean
) / problem2_train_std

problem2_X_test_standardized = (
    problem2_X_test - problem2_train_mean
) / problem2_train_std
# Part 3: Linear regression

model2 = LinearRegression()
model2.fit(
    problem2_X_train_standardized,
    problem2_y_train
)

problem2_beta = np.concatenate(
    ([model2.intercept_], model2.coef_)
).astype(float)

problem2_y_pred_test = model2.predict(
    problem2_X_test_standardized
)

problem2_residuals_test = (
    problem2_y_test - problem2_y_pred_test
)

# Part 4: Test metrics and baseline

problem2_mse_test = float(
    mean_squared_error(
        problem2_y_test,
        problem2_y_pred_test
    )
)

problem2_mae_test = float(
    mean_absolute_error(
        problem2_y_test,
        problem2_y_pred_test
    )
)

problem2_r2_test = float(
    r2_score(
        problem2_y_test,
        problem2_y_pred_test
    )
)

# Baseline: always predict the training-set mean mpg.
baseline_prediction = np.full(
    problem2_y_test.shape,
    problem2_y_train.mean(),
    dtype=float
)

problem2_baseline_mse_test = float(
    mean_squared_error(
        problem2_y_test,
        baseline_prediction
    )
)

problem2_model_beats_baseline = bool(
    problem2_mse_test < problem2_baseline_mse_test
)
old_p = 0.4
new_p = 0.5
alpha = 0.05
n_values = [10, 100, 1000, 10000]
R = 5000

def hoeffding_interval(data, alpha=0.05):
    n = data.shape[1]
    mean = np.mean(data, axis=1)
    radius = np.sqrt(np.log(2 / alpha) / (2 * n))
    lower = np.maximum(0, mean - radius)
    upper = np.minimum(1, mean + radius)
    return lower, upper

# Part 1
# Generate Bernoulli samples using new_p.
# Hint: data = rng.binomial(1, new_p, size=n)
results = []
for n in n_values:
    data = rng.binomial(1, new_p, size=[R,n])
    # Part 2 Construct a Hoeffding confidence interval.
    lower, upper = hoeffding_interval(data, alpha=alpha)
    # Part 3 Check whether the interval contains old_p.
    contains_old = (lower <= old_p)&(old_p<= upper) 
    coverage_old = np.mean(contains_old)
    # Part 4 contains_new = lower <= new_p <= upper
    contains_new = (lower <= new_p)&(new_p<= upper) 
    coverage_new = np.mean(contains_new)
    # Part 5 Summarize the simulation results in a DataFrame.
    results.append({
        "n":n,
        "coverage_old": coverage_old,
        "coverage_new": coverage_new
    })

df = pd.DataFrame(results)
print(df)


#Explain the difference: The interval should contain old_p less often when `n` becomes large, because the data are generated from new_p = 0.5

print(lower)
print(upper)
# Part 5: 95% Hoeffding interval for clipped absolute error

clipped_abs_residuals = np.clip(
    np.abs(problem2_residuals_test),
    0,
    50
)

sample_mean = float(np.mean(clipped_abs_residuals))
n_test = clipped_abs_residuals.size

# For X in [0, 50], the two-sided Hoeffding bound is
# 2 * exp(-2*n*epsilon^2 / 50^2) <= alpha.
alpha = 0.05
bound_width = 50.0

epsilon = bound_width * np.sqrt(
    np.log(2 / alpha) / (2 * n_test)
)

problem2_lower_bound = float(
    np.clip(sample_mean - epsilon, 0, 50)
)

problem2_upper_bound = float(
    np.clip(sample_mean + epsilon, 0, 50)
)
---
#### Local Test for Exam vB, PROBLEM 2
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.

# Optional local format checks for Problem 2. These checks do not prove correctness.
import numpy as np

try:
    assert isinstance(problem2_X, np.ndarray)
    assert isinstance(problem2_y, np.ndarray)
    assert problem2_X.shape[0] == problem2_y.shape[0]
    assert problem2_X.shape[1] == 6
    assert problem2_split_index == int(0.8 * problem2_X.shape[0])
    assert problem2_X_train.shape[0] == problem2_split_index
    assert problem2_X_test.shape[0] == problem2_X.shape[0] - problem2_split_index
    assert problem2_y_train.shape[0] == problem2_X_train.shape[0]
    assert problem2_y_test.shape[0] == problem2_X_test.shape[0]
    assert problem2_X_train_standardized.shape == problem2_X_train.shape
    assert problem2_X_test_standardized.shape == problem2_X_test.shape
    assert problem2_beta.shape == (7,)
    print("Problem 2 format checks passed.")
except Exception as error:
    print("Problem 2 format check failed:", error)

---
```

## Exam vB, PROBLEM 3
Maximum Points = 14

This problem is about modelling warehouse package movement as a finite homogeneous Markov chain.

The file `data/warehouse_transitions.csv` contains observed transitions between five zones:

`Dock`, `Sort`, `Storage`, `Packing`, `Dispatch`.

Use this exact state order whenever you create vectors or matrices.

1. **[3p] Estimate transition matrix.** Load the transition data and estimate the transition matrix by maximum likelihood. Store it in `problem3_transition_matrix` as a 5 by 5 row-stochastic NumPy array, where entry `(i, j)` is the estimated probability of moving from state `i` to state `j`.

2. **[2p] Four-step probability.** Starting from `Dock`, compute the probability of being in `Dispatch` after exactly 4 steps and store it in `problem3_prob_dispatch_after_4_from_dock`.

3. **[2p] Simulation.** Starting from `Dock`, simulate 20000 chains for 8 steps using `np.random.default_rng(20260616)` and the transition probabilities in `problem3_transition_matrix`. Store the empirical distribution after 8 steps in `problem3_simulated_distribution_after_8` as a length-5 probability vector in the state order above.

4. **[2p] Chain structure.** Decide whether the estimated chain is irreducible and aperiodic. Store Boolean answers in `problem3_is_irreducible` and `problem3_is_aperiodic`.

5. **[2p] Stationary distribution.** Compute a stationary distribution for the estimated transition matrix and store it in `problem3_stationary_distribution` as a length-5 probability vector in the state order above. In the markdown cell below, briefly explain what the stationary distribution means in this warehouse context.

6. **[3p] Hitting time.** Compute the expected number of steps to hit `Dispatch` for the first time when starting from `Dock`. Store it in `problem3_expected_steps_to_dispatch_from_dock`. An exact computation gives full credit; a sufficiently accurate simulation estimate can receive partial credit.

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
transition_data = pd.read_csv('data/warehouse_transitions.csv', encoding='latin1')
transition_data.head()

import pandas as pd
import numpy as np

def estimate_transition_matrix(data, state_col):
    
    # Drop missing values in the state column
    data = transition_data.dropna(subset=[state_col])

    # Ensure the column is treated as categorical
    states = pd.Categorical(data[state_col])
    state_labels = states.categories
    n_states = len(state_labels)

    # Initialize count matrix
    count_matrix = np.zeros((n_states, n_states), dtype=int)

    # Count transitions
    for (current_state, next_state) in zip(states[:-1], states[1:]):
        count_matrix[current_state, next_state] += 1

    # Apply MLE: P_ij = count(i→j) / sum(count(i→k))
    with np.errstate(divide='ignore', invalid='ignore'):
        transition_matrix = count_matrix / count_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.nan_to_num(transition_matrix)  # Replace NaN with 0

    return pd.DataFrame(transition_matrix, index=state_labels, columns=state_labels)


if __name__ == "__main__":
    # Example: Load dataset
    # The dataset must have a column with sequential states (e.g., 'state')
    try:
        df = pd.read_csv("data/warehouse_transitions.csv") 
    except FileNotFoundError:
        print("Error: dataset.csv not found. Please provide a valid file path.")
        exit(1)

    # Validate column existence
    if 'step' not in df.columns:
        print("Error: The dataset must contain a 'state' column.")
        exit(1)

    # Estimate transition matrix
    transition_df = estimate_transition_matrix(df, 'step')

    print("\nEstimated Transition Matrix (MLE):")
    print(transition_df.round(2))


Dock_to_sort = transition_data[
    (transition_data["from_zone"] != "Dock" ) & (transition_data["to_zone"] != "Sort")
].loc[:, "step"]

print(Dock_to_sort)

plt.subplots(figsize=(12, 8))
plt.hist(numbil0_2008, bins=30)
plt.xlim(left=0)
plt.grid()
plt.xlabel("Number of billionaires in 2008")
plt.ylabel("Count")
plt.show()
Dock_to_sort1 = transition_data[
    (transition_data["from_zone"] != "Sort" ) & (transition_data["to_zone"] != "Storage")
].loc[:, "step"]

print(Dock_to_sort1)

plt.subplots(figsize=(12, 8))
plt.hist(numbil0_2008, bins=30)
plt.xlim(left=0)
plt.grid()
plt.xlabel("Number of billionaires in 2008")
plt.ylabel("Count")
plt.show()
Dock_to_sort2 = transition_data[
    (transition_data["from_zone"] != "Storage" ) & (transition_data["to_zone"] != "Packing")
].loc[:, "step"]

print(Dock_to_sort2)

plt.subplots(figsize=(12, 8))
plt.hist(numbil0_2008, bins=30)
plt.xlim(left=0)
plt.grid()
plt.xlabel("Number of billionaires in 2008")
plt.ylabel("Count")
plt.show()
Dock_to_sort3 = transition_data[
    (transition_data["from_zone"] != "Packing" ) & (transition_data["to_zone"] != "Dispatch")
].loc[:, "step"]

print(Dock_to_sort3)

plt.subplots(figsize=(12, 8))
plt.hist(numbil0_2008, bins=30)
plt.xlim(left=0)
plt.grid()
plt.xlabel("Number of billionaires in 2008")
plt.ylabel("Count")
plt.show()
P = [[0.12  ,0.88  ,0.00  ,0.00  ,0.00  ,0.00],  
[0.16,  0.00,  0.84 , 0.00,  0.00,  0.00  ],
[0.18 , 0.00 , 0.00,  0.82,  0.00 , 0.00  ],
[0.18,   0.00  ,0.00 , 0.00  ,0.82,  0.00],
[0.24,   0.00 , 0.00 , 0.00 , 0.00,  0.76],
[0.30 , 0.00 , 0.00,  0.00,  0.00,  0.00 ],
[ 0.52,  0.00 , 0.00,  0.00 , 0.00  ,0.00],
[1.00  , 0.00 , 0.00,  0.00,  0.00  ,0.00 ]]

print(P)

def is_irreducible(P, tol=1e-15):
    # Check irreducibility using reachability on the directed graph.
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    adj = (P > tol)

    for start in range(n):
        seen = {start}
        stack = [start]
        while stack:
            i = stack.pop()
            for j in range(n):
                if adj[i, j] and j not in seen:
                    seen.add(j)
                    stack.append(j)
        if len(seen) != n:
            return False
    return True
def periods(P, max_power=200, tol=1e-15):
    # Compute state periods d(i) by collecting return times up to max_power.
    P = np.asarray(P, dtype=float)
    n = P.shape[0]

    returns = [[] for _ in range(n)]
    Pk = np.eye(n)

    for k in range(1, max_power + 1):
        Pk = Pk @ P
        diag = np.diag(Pk)
        for i in range(n):
            if diag[i] > tol:
                returns[i].append(k)

    def gcd_list(lst):
        from math import gcd
        g = lst[0]
        for x in lst[1:]:
            g = gcd(g, x)
        return g

    out = np.zeros(n, dtype=object)
    for i in range(n):
        out[i] = None if len(returns[i]) == 0 else gcd_list(returns[i])
    return out

def is_aperiodic(P):
    # Aperiodic if all states have period 1 (assuming computation found returns).
    per = periods(P)
    if any(p is None for p in per):
        return False
    return all(int(p) == 1 for p in per)
def stationary_distribution(P):
    # Solve pi = pi P with sum(pi)=1 using a linear system.
    P = np.asarray(P, dtype=float)
    n = P.shape[0]

    A = P.T - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0

    pi = np.linalg.solve(A, b)
    pi = np.clip(pi, 0.0, 1.0)
    pi = pi / pi.sum()
    return pi

    
def is_reversible(P, pi=None, tol=1e-8):
    # Check detailed balance pi_i P_ij == pi_j P_ji.
    P = np.asarray(P, dtype=float)
    if pi is None:
        pi = stationary_distribution(P)

    lhs = pi.reshape(-1, 1) * P
    rhs = (pi.reshape(1, -1) * P.T)


R = 5000
target_state = 2
max_steps = 1000


hitting_times = []

for r in range(R):
    current_state = 0
    t = 0

    while current_state != target_state and t < max_steps:
            current_state = rng.choice([0,1,2], p=P[current_state])
            t += 1

    hitting_times.append(t)


hitting_times = np.array(hitting_times)

mean_hitting_time = np.mean(hitting_times)
print(mean_hitting_time)

plt.hist(hitting_times, bins=30)
plt.xlabel("Hitting time")
plt.ylabel("Frequency")
plt.title("Hitting time to Rainy state")
plt.grid()
plt.show()
# Part 1: Estimate the MLE transition matrix

problem3_states = [
    "Dock", "Sort", "Storage", "Packing", "Dispatch"
]

transition_df = pd.read_csv(
    "data/warehouse_transitions.csv",
    encoding="latin1"
)

state_to_index = {
    state: i for i, state in enumerate(problem3_states)
}

counts = np.zeros((5, 5), dtype=float)

for _, row in transition_df.iterrows():
    i = state_to_index[row["from_zone"]]
    j = state_to_index[row["to_zone"]]
    counts[i, j] += 1

problem3_transition_matrix = (
    counts / counts.sum(axis=1, keepdims=True)
)
# Part 2: Probability of being in Dispatch after exactly 4 steps

P4 = np.linalg.matrix_power(
    problem3_transition_matrix,
    4
)

# Dock = 0, Dispatch = 4.
problem3_prob_dispatch_after_4_from_dock = float(
    P4[0, 4]
)
# Part 3: Simulate 20,000 chains for 8 steps

rng3 = np.random.default_rng(20260616)

R = 20000
steps = 8

# All chains start in Dock (state 0).
current_states = np.zeros(R, dtype=int)

for _ in range(steps):
    next_states = np.empty(R, dtype=int)

    for state in range(5):
        mask = current_states == state
        number = np.sum(mask)

        if number > 0:
            next_states[mask] = rng3.choice(
                5,
                size=number,
                p=problem3_transition_matrix[state]
            )

    current_states = next_states

problem3_simulated_distribution_after_8 = (
    np.bincount(current_states, minlength=5).astype(float) / R
)
# Part 4: Irreducibility and aperiodicity

P = problem3_transition_matrix
adjacency = P > 0

def reachable_from(start, adjacency_matrix):
    seen = {start}
    stack = [start]

    while stack:
        i = stack.pop()
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] and j not in seen:
                seen.add(j)
                stack.append(j)

    return seen

problem3_is_irreducible = bool(
    all(
        len(reachable_from(i, adjacency)) == 5
        for i in range(5)
    )
)

# For an irreducible finite Markov chain, a positive self-loop
# is sufficient to establish aperiodicity.
if problem3_is_irreducible:
    problem3_is_aperiodic = bool(
        np.any(np.diag(P) > 0)
    )
else:
    # General fallback: calculate each state's period.
    from math import gcd

    def state_period(P, start, max_power=100):
        Pk = np.eye(P.shape[0])
        return_times = []

        for k in range(1, max_power + 1):
            Pk = Pk @ P
            if Pk[start, start] > 1e-12:
                return_times.append(k)

        if not return_times:
            return None

        g = return_times[0]
        for k in return_times[1:]:
            g = gcd(g, k)
        return g

    state_periods = [
        state_period(P, i)
        for i in range(5)
    ]

    problem3_is_aperiodic = bool(
        all(period == 1 for period in state_periods)
    )
# Part 5: Stationary distribution

# Solve pi P = pi together with sum(pi) = 1.
A = problem3_transition_matrix.T - np.eye(5)
b = np.zeros(5)

A[-1, :] = 1.0
b[-1] = 1.0

problem3_stationary_distribution = np.linalg.solve(A, b)
problem3_stationary_distribution = np.asarray(
    problem3_stationary_distribution,
    dtype=float
)

# Remove tiny numerical normalization error.
problem3_stationary_distribution /= (
    problem3_stationary_distribution.sum()
)
## Free text answer for Part 5

The stationary distribution describes the long-run fraction of time that the warehouse movement process spends in each zone, assuming the Markov chain has the required long-run convergence properties. For example, a stationary probability of 0.30 for a zone means that, over a very long sequence of package movements, approximately 30% of the visits are expected to be in that zone.
# Part 6: Exact expected hitting time to Dispatch

# Dispatch is state 4. Let h_i be the expected number of
# steps to first reach Dispatch starting from state i.
# h_4 = 0, and for i != 4:
# h_i = 1 + sum_{j != 4} P[i,j] h_j.
# Thus (I - Q)h = 1 for the non-target states.

target = 4
transient_states = [0, 1, 2, 3]

Q = problem3_transition_matrix[
    np.ix_(transient_states, transient_states)
]

h_transient = np.linalg.solve(
    np.eye(len(transient_states)) - Q,
    np.ones(len(transient_states))
)

h_all = np.zeros(5)
h_all[transient_states] = h_transient

problem3_expected_steps_to_dispatch_from_dock = float(
    h_all[0]
)
---
#### Local Test for Exam vB, PROBLEM 3
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.

# Optional local format checks for Problem 3. These checks do not prove correctness.
import numpy as np

try:
    assert problem3_transition_matrix.shape == (5, 5)
    assert np.allclose(np.sum(problem3_transition_matrix, axis=1), 1, atol=2e-4)
    assert problem3_simulated_distribution_after_8.shape == (5,)
    assert np.all(problem3_simulated_distribution_after_8 >= -1e-12)
    assert abs(np.sum(problem3_simulated_distribution_after_8) - 1) < 2e-4
    assert problem3_stationary_distribution.shape == (5,)
    assert np.all(problem3_stationary_distribution >= -1e-12)
    assert abs(np.sum(problem3_stationary_distribution) - 1) < 2e-4
    print("Problem 3 format checks passed.")
except Exception as error:
    print("Problem 3 format check failed:", error)
