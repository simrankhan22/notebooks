
You are given the "Abalone" dataset found in `data/abalone.csv`, which contains physical measurements of abalone (a type of sea shells) and the age of the abalone measured in **rings** (the number of rings in the shell) [https://en.wikipedia.org/wiki/Abalone](https://en.wikipedia.org/wiki/Abalone). Your task is to train a `linear regression` model to predict the age (Rings) of an abalone based on its physical measurements.

To evaluate your model, you will split the dataset into a training set and a testing set. You will use the training set to train your model, and the testing set to evaluate its performance.

1. Load the data into a pandas dataframe `problem2_df`. Based on the column names, figure out what are the features and the target and fill in the answer in the correct cell below. [2p]
2. Split the data into train and test. [2p]
3. Train the model. [1p]
4. On the test set, evaluate the model by computing the mean absolute error and plot the empirical distribution function of the residual with confidence bands (i.e. using the DKW inequality and 95% confidence). Hint: you can use the function `plotEDF,makeEDF` combo from `Utils.py` that we have used numerous times, which also contains the option to have confidence bands. [3p]
5. Provide a scatter plot where the x-axis corresponds to the predicted value and the y-axis is the true value, do this over the test set. [2p]
6. Reason about the performance, for instance, is the value of the mean absolute error good/bad and what do you think about the scatter plot in point 5? [3p]

```{python}
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
# Part 1
# clean column names if they have BOM or extra whitespace
problem2_df.columns = problem2_df.columns.str.replace('\ufeff', '', regex=False).str.strip()
# Fill in the features as a list of strings of the names of the columns

problem2_features = [['Length', 'Diameter', 'Height','Whole weight','Shucked weight','Viscera weight', 'Shell weight']]
print("Features:", problem2_features)
# Fill in the target as a string with the correct column name

problem2_target = " Rings"
print("Target:", problem2_target)



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




# Part 3
# Include the necessary imports
from sklearn.linear_model import LinearRegression

# Initialize your linear regression model
problem2_model = LinearRegression()

# Train your model on the training data
problem2_model.fit(problem2_X_train, problem2_y_train)




# Part 4

# Evaluate the model by computing the mean absolute error 
from sklearn.metrics import mean_absolute_error

# Predict on the test set
y_pred = problem2_model.predict(problem2_X_test)

# Evaluate the model by computing the mean absolute error
problem2_mae = mean_absolute_error(problem2_y_test, y_pred)

print('mae:', problem2_mae)


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



```


