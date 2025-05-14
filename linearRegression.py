import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Credit Risk Benchmark Dataset.csv")
data = data[["age", "debt_ratio"]]

def mean_squared_error(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].iloc[0]
        y = points.iloc[i].iloc[1]
        total_error += (y - (m * x + b))**2
    return total_error / len(points)

def gradient_descent(m, b, points, l):
    n = len(points)
    m_gradient = 0
    b_gradient = 0

    for i in range(n):
        x = points.iloc[i].iloc[0]
        y = points.iloc[i].iloc[1]

        m_gradient += x * (y - (m * x + b))
        b_gradient += y - (m * x + b)

    m_gradient *= -2/n
    b_gradient *= -2/n

    m -= m_gradient * l
    b -= b_gradient * l

    return m, b

m = 0 
b = 0
l = 0.0001
epochs = 100

for i in range(epochs):
    m, b  = gradient_descent(m, b, data, l)
    print(m, b)

plt.scatter(data.age, data.debt_ratio, color="black")
plt.show()