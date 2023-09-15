"""Predicts the pattern of a quadratic set of data by modeling a parabola to fit the data"""
import numpy as np
import matplotlib.pyplot as plt


nrdata = 15
x_values = np.random.randn(nrdata)*2
y_values = np.random.randn() * x_values ** 2 + np.random.randn() * x_values + np.random.randn()
y_values = y_values + 0.4 * np.random.randn(nrdata)

theta = np.random.randn(3,1)


epochs = 900  
learning_rate = 0.01
len_x_values = len(x_values)

for epoch in range(epochs):

    theta_gradients = np.zeros((3,1))

    for i in range(3):
        theta_gradients[i] = sum((label - theta[2] * value ** 2 - theta[1] * value - theta[0]) * (-value**(i)) 
                             for value,label in zip(x_values,y_values))/len_x_values

    for i in range(3):
        theta[i] = theta[i] - learning_rate * theta_gradients[i]


x_valuesp = np.arange(-5, 5, 0.1)
y_valuesp = theta[0] + theta[1] * x_valuesp + theta[2] * x_valuesp ** 2

plt.figure()
plt.plot(x_values, y_values, 'bo')
plt.plot(x_valuesp, y_valuesp, 'r-')
plt.title("Learned function")
