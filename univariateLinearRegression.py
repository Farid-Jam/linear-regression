import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/univariate_data.csv")
data = data[["Hours", "Scores"]]

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
epochs = 1000

for i in range(epochs):
    m, b  = gradient_descent(m, b, data, l)
    print(m, b)
    print(mean_squared_error(m, b, data))

plt.scatter(data.Hours, data.Scores, color="black")

x_vals = data.Hours
y_vals = m * x_vals + b
plt.plot(x_vals, y_vals, color="red", label="Regression Line")

plt.legend()
plt.show()