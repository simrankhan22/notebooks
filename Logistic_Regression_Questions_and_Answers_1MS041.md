# 1MS041 Data Science --- Logistic Regression Questions & Answers

> **Source-based study file.** This document collects the
> logistic-regression and binary-classification questions, answers,
> tasks, formulas, code patterns, exam problems, and numerical results
> supported by the uploaded course material. It does not add unrelated
> regression topics.

## Contents

1.  Logistic regression fundamentals
2.  Day 5 --- Battery Health Classification
3.  Day 5 --- Threshold and Expected Cost
4.  Day 5 --- Confusion Matrix and Metrics
5.  Day 5 --- Review Questions and Mini Exam
6.  Assignment 3, Problem 3 --- ProportionalSpam
7.  Exam 2024, Problem 2 --- Logistic Regression: Calibration &
    Predictive Intervals
8.  Assignment 4 --- Text Classification Pipeline
9.  Important Python/code patterns
10. Exam revision questions
11. Logistic regression memory sheet

------------------------------------------------------------------------

# 1. Logistic regression fundamentals

## Q1. What does logistic regression model?

**Answer:** Logistic regression models the probability that an
observation belongs to a particular class.

The supplied Day 5 review states this directly.

------------------------------------------------------------------------

## Q2. What type of problem is logistic regression used for in the supplied material?

**Answer:** Binary classification.

The Day 5 material focuses on: - binary classification, - logistic
regression, - train/test split, - probabilities and thresholds, -
confusion matrix, - precision, - recall, - F1, - expected cost.

------------------------------------------------------------------------

## Q3. What is the logistic function?

**Answer:**

$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

The supplied Battery Health problem uses:

``` python
prob_unhealthy = 1 / (1 + np.exp(-z))
```

This converts the linear score `z` into a probability between 0 and 1.

------------------------------------------------------------------------

## Q4. What is the linear score in logistic regression?

**Answer:** The supplied `ProportionalSpam` implementation uses:

$$
z_i = w_0+x_i\cdot w.
$$

In code:

``` python
z = np.dot(X, coeffs[1:]) + coeffs[0]
```

The first coefficient is the intercept and the remaining coefficients
correspond to the features.

------------------------------------------------------------------------

## Q5. How is a probability converted into a class label?

**Answer:** Apply a classification threshold.

For example:

``` python
y_pred = (y_proba >= threshold).astype(int)
```

With threshold 0.5, probabilities at least 0.5 are classified as class
1.

------------------------------------------------------------------------

## Q6. What is the difference between a prediction probability and a prediction label?

**Answer:** A prediction probability is the estimated probability of a
class, while a prediction label is the final class chosen after applying
a threshold.

------------------------------------------------------------------------

## Q7. What does a classification threshold do?

**Answer:** It converts predicted probabilities into class labels.

------------------------------------------------------------------------

# 2. Day 5 --- Battery Health Classification

## Q8. What is the Battery Health Classification problem?

**Answer:** The task is to classify whether a battery is unhealthy.

Features:

``` text
age_months
cycles
avg_temperature
```

Target:

``` text
unhealthy = 1  → high failure risk
unhealthy = 0  → normal
```

------------------------------------------------------------------------

## Q9. What are the tasks in the Battery Health problem?

**Answer:**

1.  Generate synthetic data.
2.  Create target probabilities using a logistic formula.
3.  Generate the binary target using `rng.binomial`.
4.  Split the data.
5.  Fit logistic regression.
6.  Predict probabilities.
7.  Predict labels using threshold 0.5.
8.  Calculate accuracy, precision, recall and F1.

------------------------------------------------------------------------

## Q10. How was the synthetic Battery Health data generated?

**Answer:**

``` python
n = 2000

age_months = rng.integers(1, 72, size=n)
cycles = rng.integers(50, 1500, size=n)
avg_temperature = rng.normal(30, 6, size=n)
```

------------------------------------------------------------------------

## Q11. What logistic score was used for battery failure risk?

**Answer:**

$$
z =
0.04\,\text{age\_months}
+
0.003\,\text{cycles}
+
0.08\,\text{avg\_temperature}
-8.
$$

The supplied code is:

``` python
z = (
    0.04 * age_months
    + 0.003 * cycles
    + 0.08 * avg_temperature
    - 8
)
```

The problem's hint says that higher age, cycles and temperature should
increase risk.

------------------------------------------------------------------------

## Q12. How was the unhealthy probability calculated?

**Answer:**

``` python
prob_unhealthy = 1 / (1 + np.exp(-z))
```

This is the logistic function applied to the linear score.

------------------------------------------------------------------------

## Q13. How was the binary target generated?

**Answer:**

``` python
unhealthy = rng.binomial(
    1,
    prob_unhealthy,
    size=n
)
```

------------------------------------------------------------------------

## Q14. How was the feature matrix and target selected?

**Answer:**

``` python
X = df[
    ["age_months", "cycles", "avg_temperature"]
]

y = df["unhealthy"]
```

------------------------------------------------------------------------

## Q15. What train/test split was used?

**Answer:**

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

So 20% of the data is used as the test set.

------------------------------------------------------------------------

## Q16. How was logistic regression fitted in the Battery Health problem?

**Answer:**

``` python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

------------------------------------------------------------------------

## Q17. How do you get the predicted probability for class 1?

**Answer:**

``` python
y_proba = model.predict_proba(X_test)[:, 1]
```

The `[:, 1]` selects the probability of class 1.

------------------------------------------------------------------------

## Q18. How do you make predictions using threshold 0.5?

**Answer:**

``` python
threshold = 0.5

y_pred = (
    y_proba >= threshold
).astype(int)
```

------------------------------------------------------------------------

# 3. Day 5 --- Threshold and Expected Cost

## Q19. What costs were assigned to false negatives and false positives?

**Answer:**

``` text
False negative cost = 10
False positive cost = 2
```

The supplied problem deliberately makes false negatives more expensive.

------------------------------------------------------------------------

## Q20. What is a false negative?

**Answer:** A false negative occurs when the model predicts negative but
the true class is positive.

In Boolean form:

``` python
(y_true == 1) & (y_pred == 0)
```

------------------------------------------------------------------------

## Q21. What is a false positive?

**Answer:** A false positive occurs when the model predicts positive but
the true class is negative.

In Boolean form:

``` python
(y_true == 0) & (y_pred == 1)
```

------------------------------------------------------------------------

## Q22. What is the classification-cost formula used in the Day 5 problem?

**Answer:**

$$
\text{total cost}
=
2(\text{false positives})
+
10(\text{false negatives}).
$$

Average cost is:

$$
\text{average cost}
=
\frac{\text{total cost}}{n}.
$$

The supplied function is:

``` python
false_negative_cost = 10
false_positive_cost = 2

def classification_cost(
    y_true,
    y_proba,
    threshold
):
    y_pred = (
        y_proba >= threshold
    ).astype(int)

    false_negative = np.sum(
        (y_true == 1) &
        (y_pred == 0)
    )

    false_positive = np.sum(
        (y_true == 0) &
        (y_pred == 1)
    )

    total_cost = (
        false_positive_cost * false_positive
        + false_negative_cost * false_negative
    )

    avg_cost = total_cost / len(y_true)

    return avg_cost
```

------------------------------------------------------------------------

## Q23. How were thresholds tested?

**Answer:**

``` python
thresholds = np.linspace(
    0,
    1,
    101
)
```

This tests thresholds from 0.00 to 1.00 in increments of 0.01.

------------------------------------------------------------------------

## Q24. How was the optimal threshold found?

**Answer:**

``` python
costs = []

for t in thresholds:
    cost = classification_cost(
        y_test,
        y_proba,
        t
    )
    costs.append(cost)

costs = np.array(costs)

best_idx = np.argmin(costs)

optimal_threshold = thresholds[best_idx]
optimal_cost = costs[best_idx]
```

------------------------------------------------------------------------

## Q25. What was the supplied optimal threshold?

**Answer:**

``` text
Optimal threshold: 0.16
Optimal cost: 0.795
```

These are the numerical results shown in the Day 5 material.

------------------------------------------------------------------------

## Q26. Why was a threshold lower than 0.5 preferred in this problem?

**Answer:** False negatives cost 10 while false positives cost only 2. A
lower threshold makes it easier to classify an observation as positive,
which increases recall and reduces missed positive cases.

The supplied answer says that when false negatives are expensive, a
smaller classification threshold is preferable because it increases
recall and reduces the number of missed positive cases. Although
precision decreases, the overall expected cost becomes lower.

------------------------------------------------------------------------

# 4. Day 5 --- Confusion Matrix and Metrics

## Q27. What does a confusion matrix contain?

**Answer:** The supplied material describes the output as:

``` text
TN  FP
FN  TP
```

where: - TN = true negative - FP = false positive - FN = false
negative - TP = true positive

------------------------------------------------------------------------

## Q28. What tasks were required in the Confusion Matrix Comparison problem?

**Answer:**

1.  Calculate the confusion matrix for threshold 0.5.
2.  Calculate the confusion matrix for the optimal threshold.
3.  Compare precision and recall.
4.  Explain which threshold is better when false negatives are
    expensive.

------------------------------------------------------------------------

## Q29. How do you calculate a confusion matrix in scikit-learn?

**Answer:**

``` python
confusion_matrix(
    y_test,
    y_pred
)
```

------------------------------------------------------------------------

## Q30. What is precision?

**Answer:** In the supplied Battery Health hints, precision answers:

> When the model predicts 1, how often is it correct?

The code is:

``` python
precision_score(
    y_test,
    y_pred,
    pos_label=1
)
```

------------------------------------------------------------------------

## Q31. What is recall?

**Answer:** Recall answers:

> From all true 1 cases, how many did the model find?

The code is:

``` python
recall_score(
    y_test,
    y_pred,
    pos_label=1
)
```

------------------------------------------------------------------------

## Q32. What is F1?

**Answer:** The Day 5 material treats F1 as one of the classification
metrics to compare at different thresholds.

The supplied code calculates it with:

``` python
f1_score(
    y_test,
    y_pred
)
```

------------------------------------------------------------------------

## Q33. What is accuracy?

**Answer:** Accuracy is one of the metrics calculated by the supplied
Day 5 code:

``` python
accuracy_score(
    y_test,
    y_pred
)
```

------------------------------------------------------------------------

## Q34. How do you calculate all metrics for a chosen threshold?

**Answer:**

``` python
def metrics_at_threshold(
    y_true,
    y_proba,
    threshold
):
    y_pred = (
        y_proba >= threshold
    ).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            y_pred
        ),
        "cost": classification_cost(
            y_true,
            y_proba,
            threshold
        ),
        "confusion_matrix":
            confusion_matrix(
                y_true,
                y_pred
            )
    }
```

------------------------------------------------------------------------

## Q35. What generally happens when the classification threshold is lowered?

**Answer:** The supplied material says a lower threshold usually
increases recall.

------------------------------------------------------------------------

## Q36. What generally happens when the classification threshold is increased?

**Answer:** The supplied material says a higher threshold usually
increases precision.

------------------------------------------------------------------------

## Q37. When is recall more important than precision?

**Answer:** When missing positive cases is costly, such as in disease
screening or fraud detection.

------------------------------------------------------------------------

## Q38. Why can threshold 0.5 be non-optimal?

**Answer:** Because the best threshold depends on the problem, class
imbalance, and the costs of different errors.

------------------------------------------------------------------------

# 5. Day 5 --- Review Questions and Mini Exam

## Q39. What conceptual questions were included in the Day 5 review?

**Answer:** The review asks:

1.  What is the difference between prediction label and prediction
    probability?
2.  What does a classification threshold do?
3.  What is a false positive?
4.  What is a false negative?
5.  When is recall more important than precision?
6.  Why can threshold 0.5 be non-optimal?
7.  What does logistic regression model?

------------------------------------------------------------------------

## Q40. What was the Day 5 mini-exam task?

**Answer:** Create a binary classification dataset, fit logistic
regression, and then:

-   calculate predicted probabilities,
-   test thresholds from 0 to 1,
-   find the best threshold by custom cost,
-   compare precision and recall at threshold 0.5 and the best
    threshold.

------------------------------------------------------------------------

# 6. Assignment 3, Problem 3 --- ProportionalSpam

## Q41. What is the Assignment 3 ProportionalSpam problem?

**Answer:** It builds a logistic/proportional model for spam versus
non-spam SMS/email data.

The model is described as:

$$
\mathbb{P}(Y=1\mid X)
=
G(\beta_0+\beta\cdot X)
$$

where $G$ is the logistic function.

The three binary features are whether the message contains:

``` text
X1 = "free"
X2 = "prize"
X3 = "win"
```

------------------------------------------------------------------------

## Q42. What were the Assignment 3 tasks?

**Answer:**

1.  Load `data/spam.csv` and create `problem3_X` and `problem3_Y`.
2.  Implement the final logistic-regression loss function inside
    `ProportionalSpam`.
3.  Train `problem3_ps` on the training data.
4.  Predict probabilities on the calibration set.
5.  Train a calibration model using `DecisionTreeRegressor`.
6.  Use the trained model and calibrator to make final predictions on
    the test set.

The Assignment 3 problem is worth 8 points.

------------------------------------------------------------------------

## Q43. What shape should `problem3_X` have?

**Answer:**

``` text
(n_texts, 3)
```

The three columns correspond to the words:

``` text
free
prize
win
```

------------------------------------------------------------------------

## Q44. What is `problem3_Y`?

**Answer:** It is a one-dimensional array:

``` text
(n_texts,)
```

with:

``` text
1 = spam
0 = not spam
```

------------------------------------------------------------------------

## Q45. What train/calibration/test split was used?

**Answer:**

``` text
Training:    40%
Calibration: 20%
Test:        40%
```

The supplied implementation performs the split in this order without
randomly shuffling the data.

------------------------------------------------------------------------

## Q46. How were the three word features created?

**Answer:**

``` python
words = [
    "free",
    "prize",
    "win"
]

n_texts = len(df_spam)

problem3_X = np.zeros(
    (n_texts, len(words)),
    dtype=int
)

for j, w in enumerate(words):
    problem3_X[:, j] = (
        df_spam[text_col]
        .str.contains(
            fr"\b{w}\b",
            case=False,
            na=False
        )
        .astype(int)
    )
```

------------------------------------------------------------------------

## Q47. How was the spam label created?

**Answer:**

``` python
problem3_Y = (
    df_spam[label_col]
    .str.lower()
    == "spam"
).astype(int).to_numpy()
```

------------------------------------------------------------------------

# 7. Logistic regression loss

## Q48. What loss function is used in the supplied ProportionalSpam model?

**Answer:** Negative log-likelihood / cross-entropy loss.

The supplied implementation is:

$$
-\frac{1}{n}
\sum_i
\left[
Y_i\log(p_i)
+
(1-Y_i)\log(1-p_i)
\right].
$$

------------------------------------------------------------------------

## Q49. How is the probability $p$ calculated?

**Answer:**

First calculate:

$$
z_i=w_0+x_i\cdot w.
$$

Then:

$$
p_i=
\frac{1}{1+e^{-z_i}}.
$$

The supplied code is:

``` python
z = np.dot(
    X,
    coeffs[1:]
) + coeffs[0]

p = 1.0 / (
    1.0 + np.exp(-z)
)
```

------------------------------------------------------------------------

## Q50. Why does the supplied code use `np.clip`?

**Answer:** To avoid taking `log(0)`.

The supplied code is:

``` python
eps = 1e-15

p = np.clip(
    p,
    eps,
    1 - eps
)
```

This keeps the probability strictly between 0 and 1.

------------------------------------------------------------------------

## Q51. What is the complete supplied loss implementation?

**Answer:**

``` python
def loss(self, X, Y, coeffs):
    z = (
        np.dot(
            X,
            coeffs[1:]
        )
        + coeffs[0]
    )

    p = 1.0 / (
        1.0 + np.exp(-z)
    )

    eps = 1e-15

    p = np.clip(
        p,
        eps,
        1 - eps
    )

    loss = -np.mean(
        Y * np.log(p)
        + (1 - Y)
        * np.log(1 - p)
    )

    return loss
```

------------------------------------------------------------------------

## Q52. How are the logistic-regression coefficients fitted in `ProportionalSpam`?

**Answer:** The supplied implementation uses `scipy.optimize.minimize`.

``` python
from scipy import optimize

opt_loss = lambda coeffs: self.loss(
    X,
    Y,
    coeffs
)

initial_arguments = np.zeros(
    shape=X.shape[1] + 1
)

self.result = optimize.minimize(
    opt_loss,
    initial_arguments,
    method='cg'
)

self.coeffs = self.result.x
```

------------------------------------------------------------------------

## Q53. What is the structure of `ProportionalSpam`?

**Answer:**

``` python
class ProportionalSpam(object):
    def __init__(self):
        self.coeffs = None
        self.result = None

    def loss(self, X, Y, coeffs):
        ...

    def fit(self, X, Y):
        ...

    def predict(self, X):
        ...
```

------------------------------------------------------------------------

## Q54. How does `ProportionalSpam.predict()` calculate probabilities?

**Answer:** The supplied code uses:

``` python
G = lambda x: np.exp(x) / (
    1 + np.exp(x)
)

return np.round(
    10 * G(
        np.dot(
            X,
            self.coeffs[1:]
        )
        + self.coeffs[0]
    )
) / 10
```

The rounding is included in the supplied solution to help with
calibration.

------------------------------------------------------------------------

## Q55. What local test value was supplied for the loss function?

**Answer:** The local test checks that:

``` python
test_instance.loss(
    np.array([
        [1,0,1],
        [0,1,1]
    ]),
    np.array([1,0]),
    np.array([
        1.2,
        0.4,
        0.3,
        0.9
    ])
)
```

is approximately:

``` text
1.2828629432232497
```

The test accepts an absolute difference below `1e-6`.

------------------------------------------------------------------------

# 8. Probability calibration

## Q56. What is the purpose of the calibration step?

**Answer:** The supplied Assignment 3 and Exam 2024 problems explicitly
use a calibration dataset to calibrate the probabilities output by the
logistic model.

------------------------------------------------------------------------

## Q57. What is `problem3_X_pred`?

**Answer:** It contains the predictions of the trained `problem3_ps`
model on the calibration data.

It must have shape:

``` text
(n_samples, 1)
```

The supplied code is:

``` python
raw_calib_pred = (
    problem3_ps.predict(
        problem3_X_calib
    )
)

problem3_X_pred = (
    raw_calib_pred.reshape(-1, 1)
)
```

------------------------------------------------------------------------

## Q58. What calibration model was used?

**Answer:** `sklearn.tree.DecisionTreeRegressor`.

The supplied model is:

``` python
from sklearn.tree import DecisionTreeRegressor

problem3_calibrator = (
    DecisionTreeRegressor(
        max_depth=3,
        random_state=0
    )
)

problem3_calibrator.fit(
    problem3_X_pred,
    problem3_Y_calib
)
```

------------------------------------------------------------------------

## Q59. What is the calibration error formula given in the material?

**Answer:**

$$
\sqrt{
\mathbb{E}
\left[
\left|
\mathbb{E}[Y\mid f(X)]
-
f(X)
\right|^2
\right]
}.
$$

This formula is explicitly given in the Assignment 3 and Exam 2024
problem statements.

------------------------------------------------------------------------

## Q60. How are calibrated test predictions produced?

**Answer:**

``` python
test_raw_pred = (
    problem3_ps.predict(
        problem3_X_test
    )
)

test_raw_pred_2d = (
    test_raw_pred.reshape(-1, 1)
)

problem3_final_predictions = (
    problem3_calibrator.predict(
        test_raw_pred_2d
    )
)
```

------------------------------------------------------------------------

# 9. Exam 2024 --- Logistic Regression: Calibration & Predictive Intervals

## Q61. What was Exam 2024 Problem 2 about?

**Answer:** It was a 13-point problem about:

-   logistic regression,
-   spam versus non-spam classification,
-   probability calibration,
-   and a 0--1 test-loss confidence interval.

The model is:

$$
\mathbb{P}(Y=1\mid X)
=
G(\beta_0+\beta\cdot X)
$$

where $G$ is the logistic function.

The features are the presence or absence of:

``` text
free
prize
win
```

------------------------------------------------------------------------

## Q62. What were the four Exam 2024 tasks?

**Answer:**

### Part 1 --- 2 points

Load `data/spam.csv`, create:

``` text
problem2_X
problem2_Y
```

and split into:

``` text
40% training
20% calibration
40% test
```

### Part 2 --- 4 points

Implement the final logistic-regression loss inside `ProportionalSpam`.

### Part 3 --- 4 points

Train `problem2_ps`, predict on the calibration set, and train a
`DecisionTreeRegressor` calibrator.

### Part 4 --- 3 points

Use the model and calibrator on the test set, compute the 0--1 test
loss, and provide a 99% confidence interval.

------------------------------------------------------------------------

## Q63. How was `problem2_X` constructed?

**Answer:** It has shape:

``` text
(n_emails, 3)
```

with columns corresponding to:

``` text
X1 = free
X2 = prize
X3 = win
```

------------------------------------------------------------------------

## Q64. How was `problem2_Y` defined?

**Answer:**

``` text
1 = spam
0 = not spam
```

The supplied code creates it with:

``` python
problem2_Y = (
    df_spam[label_col]
    .str.lower()
    == "spam"
).astype(int).to_numpy()
```

------------------------------------------------------------------------

## Q65. How was the 40/20/40 split implemented?

**Answer:**

``` python
n_train = int(
    0.4 * n_texts
)

n_calib = int(
    0.2 * n_texts
)

n_test = (
    n_texts
    - n_train
    - n_calib
)

problem2_X_train = (
    problem2_X[:n_train]
)

problem2_X_calib = (
    problem2_X[
        n_train:
        n_train + n_calib
    ]
)

problem2_X_test = (
    problem2_X[
        n_train + n_calib:
    ]
)
```

The same indexing is applied to `problem2_Y`.

------------------------------------------------------------------------

## Q66. How was the calibration model trained in Exam 2024?

**Answer:**

``` python
problem2_ps = ProportionalSpam()

problem2_ps.fit(
    problem2_X_train,
    problem2_Y_train
)

raw_calib_pred = (
    problem2_ps.predict(
        problem2_X_calib
    )
)

problem2_X_pred = (
    raw_calib_pred.reshape(-1, 1)
)

problem2_calibrator = (
    DecisionTreeRegressor(
        max_depth=3,
        random_state=0
    )
)

problem2_calibrator.fit(
    problem2_X_pred,
    problem2_Y_calib
)
```

------------------------------------------------------------------------

# 10. Exam 2024 --- Final predictions and 0--1 loss

## Q67. How are uncalibrated test probabilities obtained?

**Answer:** The supplied solution first checks whether the model has
`predict_proba`.

``` python
if hasattr(problem2_ps, "predict_proba"):
    p_uncal = (
        problem2_ps
        .predict_proba(
            problem2_X_test
        )[:, 1]
    )
else:
    p_uncal = (
        problem2_ps.predict(
            problem2_X_test
        )
    )
```

------------------------------------------------------------------------

## Q68. How are calibrated probabilities produced?

**Answer:**

``` python
problem2_X_test_pred = (
    p_uncal.reshape(-1, 1)
)

p_cal = (
    problem2_calibrator
    .predict(
        problem2_X_test_pred
    )
)

p_cal = np.clip(
    p_cal,
    0.0,
    1.0
)

problem2_final_predictions = p_cal
```

------------------------------------------------------------------------

## Q69. What threshold was used to convert final probabilities to class predictions?

**Answer:** The supplied solution uses the standard threshold:

``` python
0.5
```

with:

``` python
y_hat = (
    problem2_final_predictions >= 0.5
).astype(int)
```

------------------------------------------------------------------------

## Q70. What is 0--1 test loss?

**Answer:** It is the fraction of incorrect classifications.

The supplied code is:

``` python
problem2_01_loss = np.mean(
    y_hat != problem2_Y_test
)
```

------------------------------------------------------------------------

# 11. 99% Hoeffding confidence interval for 0--1 loss

## Q71. How was the 99% confidence interval calculated?

**Answer:** The supplied solution treats the errors as indicators:

$$
E_i =
1[\hat y_i\neq y_i],
$$

which lie in $[0,1]$.

For confidence:

$$
1-\delta=0.99,
$$

the supplied code sets:

``` python
delta = 0.01
```

and calculates:

``` python
n = len(problem2_Y_test)

epsilon = np.sqrt(
    np.log(2.0 / delta)
    / (2.0 * n)
)
```

Then:

``` python
lower = max(
    0.0,
    problem2_01_loss - epsilon
)

upper = min(
    1.0,
    problem2_01_loss + epsilon
)

problem2_interval = (
    lower,
    upper
)
```

------------------------------------------------------------------------

## Q72. Why is the interval clipped to `[0,1]`?

**Answer:** The 0--1 loss is a probability/fraction and must lie between
0 and 1. The supplied code therefore uses:

``` python
max(0.0, ...)
```

for the lower bound and:

``` python
min(1.0, ...)
```

for the upper bound.

------------------------------------------------------------------------

# 12. Assignment 4 --- Text Classification Pipeline

## Q73. What was Assignment 4 Problem 1?

**Answer:** It was a text-classification problem using:

``` text
Corona_NLP_train.csv
```

The relevant columns were:

``` text
OriginalTweet
Sentiment
```

Neutral tweets were filtered out.

------------------------------------------------------------------------

## Q74. How was the target defined in Assignment 4?

**Answer:**

$$
Y=
\begin{cases}
1 & \text{if sentiment is positive}\\
0 & \text{if sentiment is negative}
\end{cases}
$$

The supplied task explicitly says to remove `Neutral`.

------------------------------------------------------------------------

## Q75. What data split was required in Assignment 4?

**Answer:**

``` text
Train:      60%
Test:       15%
Validation: 25%
```

The supplied problem specifically says not to split randomly because the
same split was required for everyone.

------------------------------------------------------------------------

## Q76. How could text be converted into model features?

**Answer:** The supplied problem gives `CountVectorizer` as an example
of a way to convert the text into features that can be fed into a
machine-learning model.

------------------------------------------------------------------------

## Q77. What is a pipeline in the supplied Python reference?

**Answer:** A `Pipeline` chains preprocessing and a model into one
estimator.

The reference gives:

``` python
from sklearn.pipeline import Pipeline

pipe = Pipeline(
    steps=[
        ("model", clf)
    ]
)
```

The reference also lists `Pipeline` under Assignment 4.

------------------------------------------------------------------------

# 13. Important Python/code patterns

## Q78. What imports are useful for the Day 5 logistic-regression problem?

**Answer:**

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
```

------------------------------------------------------------------------

## Q79. How do you fit scikit-learn logistic regression?

**Answer:**

``` python
model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)
```

------------------------------------------------------------------------

## Q80. How do you get class-1 probabilities?

**Answer:**

``` python
y_proba = (
    model.predict_proba(X_test)[:, 1]
)
```

------------------------------------------------------------------------

## Q81. How do you convert probabilities to labels?

**Answer:**

``` python
y_pred = (
    y_proba >= threshold
).astype(int)
```

------------------------------------------------------------------------

## Q82. How do you calculate accuracy?

**Answer:**

``` python
accuracy_score(
    y_test,
    y_pred
)
```

------------------------------------------------------------------------

## Q83. How do you calculate precision?

**Answer:**

``` python
precision_score(
    y_test,
    y_pred,
    zero_division=0
)
```

------------------------------------------------------------------------

## Q84. How do you calculate recall?

**Answer:**

``` python
recall_score(
    y_test,
    y_pred,
    zero_division=0
)
```

------------------------------------------------------------------------

## Q85. How do you calculate F1?

**Answer:**

``` python
f1_score(
    y_test,
    y_pred
)
```

------------------------------------------------------------------------

## Q86. How do you calculate a confusion matrix?

**Answer:**

``` python
confusion_matrix(
    y_test,
    y_pred
)
```

------------------------------------------------------------------------

## Q87. How do you optimize a custom logistic loss?

**Answer:** The supplied course solution uses:

``` python
from scipy import optimize

res = optimize.minimize(
    fun,
    x0,
    method="BFGS"
)
```

The `ProportionalSpam` solution specifically uses:

``` python
method='cg'
```

------------------------------------------------------------------------

## Q88. What scikit-learn calibration model was used?

**Answer:**

``` python
DecisionTreeRegressor(
    max_depth=3,
    random_state=0
)
```

------------------------------------------------------------------------

# 14. Exam revision questions

## Q89. What is the most important formula for logistic regression?

**Answer:**

$$
\boxed{
p(y=1\mid x)
=
\frac{1}
{1+e^{-(\beta_0+\beta\cdot x)}}
}
$$

------------------------------------------------------------------------

## Q90. What is the most important logistic loss formula?

**Answer:**

$$
\boxed{
L
=
-\frac{1}{n}
\sum_i
\left[
y_i\log(p_i)
+
(1-y_i)\log(1-p_i)
\right]
}
$$

This is the negative log-likelihood / cross-entropy loss used by the
supplied `ProportionalSpam` implementation.

------------------------------------------------------------------------

## Q91. Why is `eps = 1e-15` used in the loss?

**Answer:** To avoid numerical problems from evaluating `log(0)`.

------------------------------------------------------------------------

## Q92. What is the difference between probability and label?

**Answer:**

``` text
Probability → continuous value between 0 and 1
Label       → class selected after applying a threshold
```

Example:

``` text
probability = 0.73
threshold = 0.5
label = 1
```

------------------------------------------------------------------------

## Q93. What happens if the threshold is lowered?

**Answer:** The supplied material says recall usually increases, because
more observations are classified as positive.

------------------------------------------------------------------------

## Q94. What happens if the threshold is raised?

**Answer:** The supplied material says precision usually increases,
because the model requires stronger evidence before predicting positive.

------------------------------------------------------------------------

## Q95. If false negatives are expensive, should the threshold usually be higher or lower?

**Answer:** Lower.

The supplied Day 5 answer explicitly says a smaller threshold is
preferable when false negatives are expensive because it increases
recall and reduces missed positive cases.

------------------------------------------------------------------------

## Q96. Why might 0.5 not be the optimal threshold?

**Answer:** Because the optimal threshold depends on: - the problem, -
class imbalance, - and the costs of different types of errors.

------------------------------------------------------------------------

## Q97. What is the difference between the training set and calibration set in the supplied spam problem?

**Answer:**

The training set is used to fit the logistic model.

The calibration set is used to learn how to adjust/calibrate the model's
predicted probabilities.

The test set is then used for final evaluation.

------------------------------------------------------------------------

## Q98. What is the supplied spam split?

**Answer:**

$$
\boxed{
40\%\text{ train},
20\%\text{ calibration},
40\%\text{ test}
}
$$

------------------------------------------------------------------------

## Q99. What are the three spam features?

**Answer:**

``` text
free
prize
win
```

Each feature is binary: presence = 1, absence = 0.

------------------------------------------------------------------------

## Q100. What does the intercept represent in the supplied model?

**Answer:** The intercept is the first coefficient, `coeffs[0]`, and is
added to the feature-weighted linear score:

``` python
z = np.dot(
    X,
    coeffs[1:]
) + coeffs[0]
```

------------------------------------------------------------------------

## Q101. What is the sequence for solving the supplied spam logistic-regression problem?

**Answer:**

``` text
1. Load spam.csv
2. Extract features: free, prize, win
3. Create binary target: spam = 1, ham = 0
4. Split into 40% train / 20% calibration / 40% test
5. Implement logistic loss
6. Fit logistic coefficients
7. Predict probabilities
8. Use calibration data to train DecisionTreeRegressor
9. Predict on test data
10. Apply calibration
11. Apply threshold 0.5
12. Calculate 0–1 test loss
13. Calculate 99% Hoeffding confidence interval
```

------------------------------------------------------------------------

# 15. Logistic Regression Memory Sheet

## Logistic model

$$
z=\beta_0+\beta\cdot x
$$

$$
\boxed{
p=\sigma(z)=\frac{1}{1+e^{-z}}
}
$$

------------------------------------------------------------------------

## Classification threshold

``` python
y_pred = (
    y_proba >= threshold
).astype(int)
```

------------------------------------------------------------------------

## Cross-entropy / negative log-likelihood

$$
\boxed{
-\frac1n
\sum_i
[
y_i\log p_i
+
(1-y_i)\log(1-p_i)
]
}
$$

------------------------------------------------------------------------

## False positive

``` text
True = 0
Predicted = 1
```

------------------------------------------------------------------------

## False negative

``` text
True = 1
Predicted = 0
```

------------------------------------------------------------------------

## Precision

``` text
Among predicted positives,
how many are actually positive?
```

------------------------------------------------------------------------

## Recall

``` text
Among actual positives,
how many did the model find?
```

------------------------------------------------------------------------

## F1

Use:

``` python
f1_score(y_true, y_pred)
```

------------------------------------------------------------------------

## Confusion matrix

``` text
TN  FP
FN  TP
```

------------------------------------------------------------------------

## Cost-sensitive thresholding

For the supplied Battery Health problem:

``` text
FN cost = 10
FP cost = 2
```

Test:

``` python
np.linspace(0, 1, 101)
```

and select the threshold with minimum average cost.

Supplied result:

``` text
optimal threshold = 0.16
optimal cost = 0.795
```

------------------------------------------------------------------------

## Spam logistic regression

Features:

``` text
free
prize
win
```

Target:

``` text
spam = 1
ham  = 0
```

Split:

``` text
40% train
20% calibration
40% test
```

------------------------------------------------------------------------

## Calibration

``` python
raw_pred = model.predict(X_calib)

X_pred = raw_pred.reshape(-1, 1)

calibrator = DecisionTreeRegressor(
    max_depth=3,
    random_state=0
)

calibrator.fit(
    X_pred,
    y_calib
)
```

------------------------------------------------------------------------

## 0--1 loss

``` python
loss = np.mean(
    y_hat != y_test
)
```

------------------------------------------------------------------------

## 99% Hoeffding interval

``` python
delta = 0.01
n = len(y_test)

epsilon = np.sqrt(
    np.log(2.0 / delta)
    / (2.0 * n)
)

lower = max(
    0.0,
    loss - epsilon
)

upper = min(
    1.0,
    loss + epsilon
)

interval = (
    lower,
    upper
)
```

------------------------------------------------------------------------

# 16. Highest-priority questions to practice before the exam

1.  **What does logistic regression model?**
2.  **Write the logistic/sigmoid function.**
3.  **Given coefficients and features, calculate the linear score and
    probability.**
4.  **Explain the difference between probability and predicted label.**
5.  **Explain what a classification threshold does.**
6.  **Calculate labels from probabilities for a given threshold.**
7.  **Define false positive and false negative.**
8.  **Explain precision and recall.**
9.  **Explain why a lower threshold increases recall.**
10. **Explain why threshold 0.5 may not be optimal.**
11. **Calculate a custom classification cost.**
12. **Search thresholds from 0 to 1 and find the minimum-cost
    threshold.**
13. **Read a confusion matrix as TN/FP/FN/TP.**
14. **Calculate accuracy, precision, recall and F1 with scikit-learn.**
15. **Implement logistic cross-entropy loss.**
16. **Explain why probabilities are clipped before taking logarithms.**
17. **Use `scipy.optimize.minimize` to fit a custom logistic model.**
18. **Create binary text features from `free`, `prize`, and `win`.**
19. **Implement the `ProportionalSpam` class.**
20. **Use the 40/20/40 train-calibration-test split.**
21. **Train a `DecisionTreeRegressor` calibrator.**
22. **Apply the calibrator to test probabilities.**
23. **Calculate 0--1 test loss.**
24. **Calculate the supplied 99% Hoeffding confidence interval.**
25. **Explain the full spam-classification pipeline from raw messages to
    calibrated test predictions.**

------------------------------------------------------------------------

# 17. Source coverage and limitations

The logistic-regression material in the uploaded files includes:

-   **Day 5 --- Logistic Regression & Classification**
    -   Battery Health Classification
    -   threshold selection
    -   expected cost
    -   confusion matrix
    -   precision
    -   recall
    -   F1
    -   review questions
    -   mini exam
-   **Assignment 3, Problem 3 --- ProportionalSpam**
    -   logistic model
    -   custom loss
    -   optimization
    -   calibration
    -   final predictions
-   **Exam 2024, Problem 2 --- Logistic regression: calibration &
    predictive intervals**
    -   spam classification
    -   40/20/40 split
    -   custom logistic loss
    -   calibration
    -   0--1 test loss
    -   99% Hoeffding interval
-   **Assignment 4, Problem 1**
    -   binary text classification
    -   sentiment target
    -   train/test/validation split
    -   text-to-feature conversion
    -   `CountVectorizer` as an example
    -   pipeline concept

The uploaded preparation reference also indexes Exam January 2022
Problem 4 as **SMS spam filtering**, but that problem is an empirical
conditional-probability/Hoeffding exercise rather than a
logistic-regression model, so it is **not included as logistic
regression** here.

Where the source material gives a method but does not expose a final
numerical answer, this document preserves the method rather than
inventing a result.
