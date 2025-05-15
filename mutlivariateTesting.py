import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multivariateLinearRegression import multivariateLinearRegression

## MULTIVARIATE LINEAR REGRESSION
# Initialize database
data = pd.read_csv("data/multivariate_data.csv")

# Clean database
data = data.replace({'yes': 1, 'no': 0}).infer_objects(copy=False)
data['sex'] = data['sex'].map({'M': 1, 'F': 0})
data['famsize'] = data['famsize'].map({'LE3': 1, 'GT3': 0})
data['school'] = data['school'].map({'GP': 1, 'MS': 0})
data['address'] = data['address'].map({'U': 1, 'R': 0})
data['Pstatus'] = data['Pstatus'].map({'A': 1, 'T': 0})
data['Mjob'] = data['Mjob'].map({'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4})
data['Fjob'] = data['Fjob'].map({'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4})
data['reason'] = data['reason'].map({'course': 0, 'home': 1, 'reputation': 2, 'other': 3})
data['guardian'] = data['guardian'].map({'mother': 0, 'father': 1, 'other': 2})
print(data)

# Prepare data for function
y = data[['G3']].to_numpy()
x = data.drop(columns=['G3']).to_numpy()
y_train = y[:350]
y_test = y[350:395]
x_train = x[:350]
x_test = x[350:395]

# Run function and print results
result, mse = multivariateLinearRegression(x_train, y_train, x_test, y_test)
print(result)
print(mse)