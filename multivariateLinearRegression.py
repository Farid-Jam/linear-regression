import numpy as np

def multivariateLinearRegression(training_x, training_y, testing_x, testing_y):
    x_train = np.hstack((np.ones((training_x.shape[0], 1)), training_x))
    y_train = training_y

    b = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train

    x_test = np.hstack((np.ones((testing_x.shape[0], 1)), testing_x))

    y_pred = x_test @ b
    residuals = testing_y - y_pred

    result = np.hstack((testing_y, y_pred))

    mse = np.mean(residuals**2)

    return result, mse