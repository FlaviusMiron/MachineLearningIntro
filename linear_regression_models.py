"""
Script for visualizing linear regressin with both batch and stochastic gradient descent. It also has a locally weighted regression model.
Data sets are already provided.
"""


import numpy as np
from matplotlib import pyplot as plt
import random

i = 1

class linear_regression():
    def __init__(self, points):
        self.points = points

        self.weigth = np.random.randn() 
        self.bias = np.random.randn()

        # Next we have some improving quality-of-life attributes that have nothing to do with the matematical models, rather they just make the plots more clear

        self.first_plot = True
        self.last_plot = False 
        self.starting_color = "#000000"

    def plot_points(self):
        """Plots the training data. It is plotted by default by each model so there is no need to call it unless you want to just see the data."""
        plt.figure()
        for x,y in self.points:
            plt.plot(x,y,marker="o",markeredgecolor="green", markerfacecolor="green")
        plt.xlabel("Data")
        plt.ylabel("Price")

    def stochastic_gradient_descent(self, learning_rate = 1):
        """
        Fits a line to the data using purely stochastic gradient descent, meaning a single point is used a signle time. Multiple iterations could be implemented
        But i find that it fits the set good enough right now.
        It is more complex than the other models in terms of code, but the matematical part is the same. The complexity comes from the fact that i added extra
        code to save each set of parameters, so that it can plot the best preforming line with yellow after training. The reason for which I do this only here 
        is because the stochastic nature of the algorithm will cause stochastic oscillascions towards convergence, therefore it is possible that a better perforimng 
        line will be found before the last line learned.
        """
        costs = []
        self.weigth_s = [] # Stores all the parameters here
        self.bias_s = []

        self.plot_points()

        i=0 # Used for plotting, has nothing to do with the algorithm
        lenght_training_data = len(self.points)

        
        for x,y in self.points:
            self.plot_linear_regression("stochastic",lenght_training_data)

            nabla_a = (-x) * (y - self.weigth * x - self.bias)
            nabla_b = (y - self.weigth * x - self.bias)

            self.weigth = self.weigth - learning_rate*nabla_a
            self.bias = self.bias - learning_rate*nabla_b

            self.weigth_s.append(self.weigth)
            self.bias_s.append(self.bias)

            cost_function =sum([(out - self.weigth * inp - self.bias)**2 for inp,out in self.points]) / (2 * lenght_training_data)

            costs.append(cost_function)

            print("Stochastic GD cost:",cost_function,"on iteration",i)

            if i == (len(self.points) -1): # Plot-related logic
                self.last_plot = True
            i+=1


        print("w=",self.weigth,"b=" ,self.bias)
        self.plot_linear_regression(id="stochastic")


        optimal_index = costs.index(min(costs)) # Get the best line that was fit throughout the training
        print("Optimal index:",optimal_index,"(yellow line)")
        optimal_a = self.weigth_s[optimal_index]
        optimal_b = self.bias_s[optimal_index]
        self.helping_plot(optimal_a,optimal_b)

    def batch_gradient_descent(self, learning_rate = 0.1, epochs=30): 
        """Fits a line to the data using batch gradient descent, meaning that on each iteration every data point will be used."""
        len_training_data = len(self.points)
        self.plot_points()

        for i in range(epochs):
            self.plot_linear_regression("batch",epochs)
            nabla_a = sum([(-x)*(y - self.weigth * x - self.bias) for x,y in self.points]) / len_training_data
            nabla_b = sum([-(y - self.weigth * x - self.bias) for x,y in self.points]) / len_training_data

            self.weigth = self.weigth - learning_rate*nabla_a
            self.bias = self.bias - learning_rate*nabla_b

            cost_function =sum([(out - self.weigth * inp - self.bias)**2 for inp,out in self.points]) / (2 * len_training_data)

            print("Batch cost:",cost_function,"on ineration",i)

            if i == (epochs - 1): # Plot related logic
                self.last_plot = True

        print("w=",self.weigth,"b=" ,self.bias)
        self.plot_linear_regression("batch",epochs)

    def local_weighted_regression(self, learning_rate = 0.1 ,prediction_point = 1 ,epochs = 30, tau = 0.5):
        """
        Fits a locally weighted line to a given set of points. prediction_point is the point around wiich the training data has the highest weights.
        The tau parameter is the attenuation of the further data points from the prediction_point, also known as bandwidth parameter
        Note that calling this method will switch to another training data set, so the "plot_points" method called before this method will generate
        different results.
        Uses batch gradient descent.
    
        """
        self.weigth = 0 # Using other parameters than random for better visualization, they can be deleted if you want random instead
        self.bias = 0

        self.points = linear_regression.weighted_regression_data()
        self.plot_points()

        training_data_length = len(points)

        for i in range(epochs):
            self.plot_linear_regression("batch",epochs)
            nabla_a = sum([(-x)*(y - self.weigth * x - self.bias)*(np.exp(-((prediction_point - x)**2)/(2*tau))) for x,y in self.points]) / training_data_length
            nabla_b = sum([-(y - self.weigth * x - self.bias)*(np.exp(-((prediction_point - x)**2)/(2*tau))) for x,y in self.points]) / training_data_length


            self.weigth = self.weigth - learning_rate*nabla_a
            self.bias = self.bias - learning_rate*nabla_b

            cost_function =sum([((out - self.weigth * inp - self.bias)**2)*(np.exp(-((prediction_point - inp)**2)/(2*tau))) for inp,out in self.points]) / (2 * training_data_length)



            print("Locally weighted cost:",cost_function,"on ineration",i)
            if i == (epochs - 1): # Plot related logic
                self.last_plot = True

        print("a=",self.weigth,"b=" ,self.bias)
        self.plot_linear_regression("batch",epochs) #plots like batch cause it works with epochs
        plt.plot(prediction_point,self.weigth * prediction_point + self.bias,marker="x",markeredgecolor="yellow", markerfacecolor="yellow",zorder=10)

    
    def normal_ecuations_cost(self):
        """
        Each linear regression algorithm has a fixed mathematical optimal parameters solution that mainimazes the cost function, known as the theoretical minima.
        It is found through the so called "normal equations", which i implemented here. I implemented this variant mainly to compare the performance of the algorithms
        to the maximum theoretical performance that could be achieved. Wether this equations should be used instead of a machine learning model depends of the lenght of
        the training data.
        """

        n = len(self.points)
        row_1 = np.ones((1,n))

        row_2 = np.array([i for i,j in self.points])

        X=np.array([row_1[0],row_2]).transpose()

        Y = np.array([j for i,j in self.points]).transpose()
        result = np.matmul(X.transpose(),X)


        inv = np.linalg.inv(result)


        result=np.matmul(inv,X.transpose())
        gradient= np.matmul(result,Y)  
        cost_function =sum([(out - gradient[1] * inp - gradient[0])**2 for inp,out in self.points]) / (2*len(self.points))
        return cost_function


    def plot_linear_regression(self,id ,epochs = None):
        """Method used by the models to plot subsequent lines."""
        global i

        self.color_values = self.generate_colors(id,epochs)
        x_axis = np.linspace(self.points[0][0]-0.5,self.points[-1][0]+0.5,100)

        regression_line = self.weigth * x_axis + self.bias
        if self.last_plot:
            plt.plot(x_axis,regression_line,color="red")
        elif self.first_plot: 
            plt.plot(x_axis,regression_line,color=self.starting_color) #starting
            self.first_plot = False
        else:
            value = self.color_values[i]
            plt.plot(x_axis,regression_line,color = value)
            i+=1

    def generate_colors(self,id,epochs):
        """Generates linearly distributed color nuances, to make the plots more pleasing."""
        if id == "batch":
            nums = np.linspace(20,255,epochs)
            colors = ["#0000"+hex(int(item))[2:].upper() for item in nums]

        elif id == "stochastic":
            nums = np.linspace(20,255,len(self.points))
            colors = ["#0000"+hex(int(item))[2:].upper() for item in nums]
        
        return colors 

    def helping_plot(self,a,b):
        """Used by the stochasting gradient descent method to plot the optimal line fit"""
        x_axis = np.linspace(self.points[0][0]-0.5,self.points[-1][0]+0.5,100)
        # plt.xlim([self.points[0][0]-0.5,self.points[-1][0]+0.5])
        # plt.ylim([self.points[0][1]-0.5,self.points[-1][1]+0.5])
        regression_line = a * x_axis + b
        plt.plot(x_axis,regression_line,color="yellow") 
      
    @staticmethod
    def weighted_regression_data():
        """Used by the locally weighted regression to load a different type of training data, as the data has to be more complex for such models."""
        rng = np.random.RandomState(0)

        n_sample = 100
        data_max, data_min = 1.4, -1.4
        len_data = (data_max - data_min)

        data = np.sort(rng.rand(n_sample) * len_data - len_data / 2)
        noise = rng.randn(n_sample) * .3
        target = data ** 3 - 0.5 * data ** 2 + noise

        points_for_wr=[]

        for point in zip(np.linspace(data_min,data_max,len(target)),target):
            points_for_wr.append(list(point))

        return points_for_wr



if __name__ == "__main__":

    points = np.array([[1.3, 2.3],[1.4, 2.6],[1.5, 2.5],[1.7, 2.9],[1.8, 2.6],[2, 3.4],[2.2, 3.6],[2.4, 3.6],[2.5, 3.7], # Data for batch and stochastic gradient descent
                       [2.6, 4],[2.7, 4.1],[2.8, 4.1],[3, 3.9],[3.2, 4.3],[3.3, 4.2],[3.4, 4.5],[3.5, 4.9],[3.6, 4.7],[3.6, 4.9],[3.7, 5]])


    model = linear_regression(points)
    #model.plot_points()

    model.stochastic_gradient_descent(learning_rate=0.05)


    #model.batch_gradient_descent(learning_rate=0.01,epochs=30)


    #model.local_weighted_regression()
    


    print("Theoretical minima:",model.normal_ecuations_cost())

