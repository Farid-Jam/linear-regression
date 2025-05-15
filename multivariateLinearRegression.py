import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/multivariate_data.csv")
data = data.replace({'yes': 1, 'no': 0})
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

y = data[['G3']].to_numpy()
x = data.drop(columns=['G3']).to_numpy()
x = np.hstack((np.ones((x.shape[0], 1)), x))

b = np.linalg.inv(x.T @ x) @ x.T @ y

y_pred = x @ b

residuals = y - y_pred

rows = y.shape

mse = np.mean(residuals**2)

result = np.hstack((y, y_pred))

print(result)
print(mse)