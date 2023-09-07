"""
Script for visualising logistic regression on a small data set, for single feature classification.
The black line repressets the initial guess sigmoid cumulatice probability distribution, and it gets bluer for each learning iteration of the algorithm.
The final sigmoid that the algorithm learns is the red one.
"""
import numpy as np
from matplotlib import pyplot as plt

class Logistic_regression():
    def __init__(self):
        self.training_data = np.array([[1, 0],[1.1, 0],[1.2, 0],[1.3, 0],[1.4, 0],
                                       [1.5, 1],[1.6, 0],[1.7, 0],[1.8, 1],[1.9, 1],
                                       [2, 0],[2.1, 1],[2.2, 1],[2.3, 1],[2.4, 1]])
        self.x_axis = np.linspace(1,2.5,20)

    def plot_points(self):  
        for x,y in self.training_data:
            plt.plot(x,y,marker="o",markeredgecolor="green", markerfacecolor="green")
        plt.xlabel("Data")
        plt.ylabel("Class")

    def maximum_likelihood(self, learning_rate, epochs):
        """Fits a sigmoid distribution to the data by miximizing the likelihood of the parameters""" 
        # self.a = np.random.randn()
        # self.b = np.random.randn()

        self.a = -1.3 #Some plott-friendly values for initial guesses
        self.b = -0.3

        self.costs = []
        self.costs.append(self.__compute_cost_function())
        colors = self.__generate_colors(epochs)

        for i in range(epochs):
        
            nabla_a = sum([(y - self.__sigmoid( self.a * x + self.b))*y for x,y in self.training_data]) 
            nabla_b = sum([(y - self.__sigmoid( self.a * x + self.b)) for x,y in self.training_data]) 
            self.a = self.a + learning_rate * nabla_a 
            self.b = self.b + learning_rate * nabla_b

            if i == epochs-1:
                plt.plot(self.x_axis, self.__sigmoid(self.a * self.x_axis + self.b),color = "red")  
            else:
                plt.plot(self.x_axis, self.__sigmoid(self.a * self.x_axis + self.b),color = colors[i])

            cost_function = self.__compute_cost_function()
            self.costs.append(cost_function)

        
    def __compute_cost_function(self):
        """
        This is actually the likelihood more than the cost, and it is derived from the statistical interpretation of the model.
        It is this function that we maximize in order for the model to learn.
        """
        return sum([y * np.log(self.__sigmoid(self.a * x + self.b)) + (1-y) * np.log(1-self.__sigmoid(self.a * x + self.b))
              for x,y in self.training_data]) * (1 / len(self.training_data))
        

    def __generate_colors(self,epochs):
        """Create linearly distributed color nouances for as many epochs there are."""
        nums = np.linspace(20,255,epochs)
        colors = ["#0000"+hex(int(item))[2:].upper() for item in nums]
        return colors  

    def __sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def plot_cost_function(self):
        plt.figure()
        plt.plot(self.costs)
        plt.xlabel("Epochs")
        plt.ylabel("Loss Function")

def main():
    model = Logistic_regression()
    model.plot_points()
    model.maximum_likelihood(0.1, 90)
    model.plot_cost_function()

if __name__ == "__main__":
    main()
