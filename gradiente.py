import numpy as np

def linear_cost(X, y, theta):
    return np.sum(((np.matmul(X, theta)) - y) ** 2) / (2 * len(X))

def linear_cost_derivate(X, y, theta):
    return np.matmul((np.matmul(X, theta) - y).T, X).T / X.shape[0]

def linear_cost_regularizado(X, y, lambdaR, theta):
    return (np.sum(((np.matmul(X, theta)) - y) ** 2) + np.sum(lambdaR * theta**2)) / (2 * len(X))

def linear_cost_derivate_regularizado(X, y, lambdaR, theta):
    return ((np.matmul((np.matmul(X, theta) - y).T, X).T) + (np.sum(lambdaR * theta))) / len(X)

def gradient_descent(
        X,
        Y,
        theta_0,
        cost,
        cost_derivate,
        alpha = 0.01,
        treshold = 0.0001,
        max_iter = 10000
    ):

    theta, i = theta_0, 0

    while np.linalg.norm(cost_derivate(X, Y, theta)) > treshold and i < max_iter:
        theta -= alpha * cost_derivate(X, Y, theta)
        i += 1

    return theta

def gradient_descent_reg(
        X,
        Y,
        theta_0,
        cost,
        cost_derivate,
        lambdaa = 0,
        alpha = 0.01,
        treshold = 0.0001,
        max_iter = 10000
    ):

    theta, i = theta_0, 0

    while np.linalg.norm(cost_derivate(X, Y, lambdaa, theta)) > treshold and i < max_iter:
        theta -= alpha * cost_derivate(X, Y, lambdaa, theta)
        i += 1

    return theta