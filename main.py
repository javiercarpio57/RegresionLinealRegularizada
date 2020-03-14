import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from gradiente import *

dataset = pd.read_csv('Admission_Predict.csv', usecols=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Chance of Admit'])
chance = dataset.iloc[:len(dataset), 6].values.reshape(-1,1)
y = chance
toefl = dataset.iloc[:len(dataset), 1].values.reshape(-1, 1)

a = np.insert(toefl**2, 0, 1, axis=1)
X = np.hstack((a, toefl**2/1000, toefl**3/1000000))
m, n = X.shape
theta_0 = np.random.rand(n, 1)

theta, costs, gradient_norms = gradient_descent (
    X,
    y,
    theta_0,
    linear_cost,
    linear_cost_derivate,
    alpha=0.0000000125,
    treshold=0.01,
    max_iter=1000000000
)

formula = ""
for i in range(len(theta) - 1,  -1, -1):
    if theta[i][0] >= 0:
        formula = " + " + str(theta[i][0]) + " * x^" + str(i) + formula
    else:
        formula = str(theta[i][0]) + " * x^" + str(i) + formula

plt.scatter(X[:,1], y, color='orange')
plt.scatter(X[:, 1], np.matmul(X, theta), color='green')
plt.title('$y=%s$'%formula)

# print(len(costs))
# plt.scatter(X[:,1], y, color='orange')
# plt.scatter(X[:, 1], np.matmul(X, theta), color='green')

# plt.plot(np.arange(len(costs)), costs)

plt.show()


# Histogramas
# Cada feature contra cada prediccion
# 