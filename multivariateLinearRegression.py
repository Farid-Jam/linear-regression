import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/multivariate_data.csv")
data = data[['age', 'studytime', 'absences', 'G1', 'G2', 'G3']]

y = data[['G3']].to_numpy()
x = data[['age', 'studytime', 'absences', 'G1', 'G2']].to_numpy()
x = np.hstack((np.ones((x.shape[0], 1)), x))
x_transpose = np.transpose(x)
x_t_times_x = np.matmul(x_transpose, x)
x_inverse = np.linalg.inv(x_t_times_x)
b = np.matmul(x_inverse, x_transpose)
b = np.matmul(b, y)

y_pred = np.matmul(x, b)

residuals = y - y_pred

rows = y.shape

mse = np.mean(residuals**2)

result = np.hstack((y, y_pred))

print(result)