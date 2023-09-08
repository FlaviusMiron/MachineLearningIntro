"""
Script that trains a single neuron using the both cross entropy cost and quadratic function.
Made to compare the 2 cost functions and show that the cross entropy avoid the learning slowdown
of the output neurons that start wrong.
Both neurons start with values close to 1, while the expected output is 0.
"""


import numpy as np
from matplotlib import pyplot as plt


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

eta = 0.15
epochs = 300

inp = 1
weight = 2
bias = 2
exp_out = 0
costs = []

print("For cross-entropy cost function:")
print("Initial weight,bias:",weight,bias,"output:",sigmoid(weight*inp + bias))

for i in range(epochs):
    z = inp * weight + bias
    a = sigmoid(z)

    cost = -(exp_out * np.log(a) + (1-exp_out)* np.log(1-a))
    costs.append(cost)
 
    nabla_w = inp * (sigmoid(z) - exp_out)
    nabla_b = sigmoid(z) - exp_out

    weight = weight - eta * nabla_w
    bias = bias - eta * nabla_b

plt.figure()
plt.plot(costs)
plt.title("Cross entropy cost")
plt.xlabel("Epochs")
plt.ylabel("Cost")
print("Final weight,bias:",weight,bias,"output:",sigmoid(weight*inp + bias))


epochs = 300

inp = 1
weight = 2
bias = 2
exp_out = 0
costs = []

print("For quadratic cost function:")
print("Initial weight,bias:",weight,bias,"output:",sigmoid(weight*inp + bias))

for i in range(epochs):
    z = inp * weight + bias
    a = sigmoid(z)

    cost = 0.5 * ((a - exp_out)**2)
    costs.append(cost)
 
    nabla_w =(a - exp_out) * sigmoid_prime(z) * inp
    nabla_b = (a - exp_out) * sigmoid_prime(z)

    weight = weight - eta * nabla_w
    bias = bias - eta * nabla_b

plt.figure()
plt.plot(costs)
plt.title("Quadratic cost")
plt.xlabel("Epochs")
plt.ylabel("Cost")
print("Final weight,bias:",weight,bias,"output:",sigmoid(weight*inp + bias))




