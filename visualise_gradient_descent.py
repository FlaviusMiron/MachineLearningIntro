"""
Script made for visualising gradiend descent optimization method, both normal and momentum based.
"""
from matplotlib import pyplot as plt
import numpy as np

class Graph():
    def __init__(self):
        self.x_axis = np.linspace(-10,10,1000)
        self.y_axis = self.test_function(self.x_axis)


    def test_function(self,x):
        #return np.sin(x)+np.sin((10.0/3.0)*x)
        return np.power(x,2)

    def test_function_prime(self,x):
        #return np.cos(x) + (10.0/3.0)*np.cos((10.0/3.0)*x)
        return 2*x

    def plot_gradient_route(self, initial_guess = None,learning_rate = 0.5 ,epochs = 30 , momentum = 0.5, plot_derivatives = False):

        plt.figure()
        plt.plot(self.x_axis,self.y_axis)
        if initial_guess == None:
            initial_guess = np.random.choice(self.x_axis)

        
        plt.plot(initial_guess,self.test_function(initial_guess),"bo")

        miu = momentum
        v0 = 0.

        current_point = initial_guess
        current_velocity = v0
        for i in range(epochs):
            if plot_derivatives:
                self.plot_derivative(current_point)

            current_gradient = self.test_function_prime(current_point)

            next_velocity = miu * current_velocity - learning_rate * current_gradient
            next_point = current_point + next_velocity

            plt.plot(next_point,self.test_function(next_point),"ro")
            current_point = next_point
            current_velocity = next_velocity

    def plot_derivative(self, x ):
        """Also plots the derivatives at each point. Might make the graph crowded."""
        reduced_x_axis = np.linspace(x-2,x+2,100)
        line_ecuation =  self.test_function_prime(x) * reduced_x_axis - x * self.test_function_prime(x) + self.test_function(x)
        plt.plot(reduced_x_axis,line_ecuation)


gradient_graph = Graph()
# Some variants provided for the y = x^2 function.
gradient_graph.plot_gradient_route(learning_rate=0.1, initial_guess=9, momentum = 0.4, epochs= 30)
gradient_graph.plot_gradient_route(learning_rate=0.1, initial_guess=9, momentum = 0., epochs= 30, plot_derivatives= True)
gradient_graph.plot_gradient_route(learning_rate=0.01, initial_guess=-9, momentum = 0., epochs= 90)












