"""
Script for visualizing a 2-feature (without the default), 2 class classification.
Inspired by Sebastian Lague's video on creating Neural Networks: https://www.youtube.com/watch?v=hfMk-kjRv4c&t=777s
Uses logistic regression to classify the 2 "fruits" based on the features.
I will also implement a neural network approach soon.

"""

import numpy as np
import matplotlib.pyplot as plt

class Logistic_Regression:
    def __init__(self, training_data = None, test_data = None):
        """
        Has some default data to run tests. Custom added data has to be of the same format:
        first 2 entries of a data element are the 2 features, while 3rd entry is the class 
        it belongs to (either 0 or 1).

        """
        if training_data is None:
            self.training_data = [
                [1,1,0],[1,2,0],[1,3,0],
                [2,0,0],[3,1,0],[4,2,0],
                [4,3,1],[4,4,1],[5,4,1],
                [5,5,1],[5,6,1],[6,6,1],
                [7,1,1],[1,10,1],[1,8,1],
                [8,8,1],[2,2,0],[0,3,0],
                [0,0,0],[0,5,0],[4,0,0],
                [2,6,1],[3,7,1],[3,5,1],
                [2,4,0],[1,6,1],[3,3,0],
                [0,6,0],[5,0,0],[0,8,1],
                [0,2.8,0],[0.8,3.8,0],[0,2,0],
                [3,9,1],[6,4,1],[3.2,5.9,1],
                [1.5,2,0],[2.8,0,0],[6,1,1],
                [3.5,4,1],[4.5,8,1],[6.5,9.5,1],
                [1.5,0,0],[2.2,0.4,0],[3.5,0.6,0],
                [1,0,0],[2,8,1],[2.5,1.6,0],
                [2.5,7,1],[3.5,5,1],[3.5,8,1],
                [1,5,0],[0.5,0.5,0],[2,2,0],
                [2.5, 1.5, 0],[7,6,1],[5,2,1],
                [0.1,0.1,0],[0.3,0.5,0],[0.5,3.2,0],
                [4,6.5,1],[4.5,9,1],[1.5,9,1],
                [5.5,4,1],[4.5,7,1],[3,8,1],
                [1.5,4,0],[3.2,2,0],[3.5,0,0]]
            
        if test_data is None:
            self.test_data = [
                              [4,6,1],[2,4.5,1],[0,8.5,1],
                              [0.5,1,0],[1.5,1,0],[6,9,1],
                              [6,2,1],[2,6.5,1],[0,6.5,0],
                              [4.5,1.8,0],[2.5,3.8,0],[5.5,6,1],
                              [1,2.2,0],[0.5,4,0],[2.8,7.3,1]]
            
        self.thetas = np.random.randn(1,3)
        self.default_parameter_test = False
        self.default_parameter_train = False


    def plot_training_data(self):
        if self.default_parameter_train:
            self.__remove_default_parameter(self.training_data)
            self.default_parameter_train = False

        for point in self.training_data:
            if point[2]:
                plt.plot(point[0], point[1], marker="o", markersize=10, 
                         markerfacecolor="green", markeredgecolor="green")
            else:
                plt.plot(point[0], point[1], marker="x", markersize=10,
                          markerfacecolor="green", markeredgecolor="green")

    def plot_test_data(self):
        if self.default_parameter_test:
            self.__remove_default_parameter(self.test_data)
            self.default_parameter_test = False

        for point in self.test_data:
            if point[2]:
                plt.plot(point[0], point[1], marker="o", markersize=10,
                          markerfacecolor="blue", markeredgecolor="blue")
            else:
                plt.plot(point[0], point[1], marker="x", markersize=10,
                          markerfacecolor="blue", markeredgecolor="blue")
                
    def logistic_regression(self, epochs = 15, learning_rate = 0.01):
        if not self.default_parameter_train:
            self.__add_default_parameter(self.training_data)
            self.default_parameter_train = True

        self.costs = []
        self.costs.append(self.__compute_quadratic_cost(self.training_data))

        nabla_thetas = np.zeros((1,3))
        
        for epoch in range(epochs):
            #Here we calculate the partial derivatives of the log-likelihood with respect to each parameter x, y and z
            nabla_thetas[0][0] = sum([ (prediction - self.__sigmoid(np.matmul(self.thetas,np.array([x,y,z]))))*(-x) 
                                for x,y,z,prediction in self.training_data])
            nabla_thetas[0][1] = sum([ (prediction - self.__sigmoid(np.matmul(self.thetas,np.array([x,y,z]))))*(-y) 
                                for x,y,z,prediction in self.training_data])
            nabla_thetas[0][2] = sum([ (prediction - self.__sigmoid(np.matmul(self.thetas,np.array([x,y,z]))))*(-z) 
                                for x,y,z,prediction in self.training_data])
           

            #Here we update the gradient descent algorithm with the partial derivatives calculated above
            self.thetas = [t-learning_rate*n_t for t,n_t in zip(self.thetas, nabla_thetas)]
            self.costs.append(self.__compute_quadratic_cost(self.training_data))




            
    def __compute_quadratic_cost(self, data):
        if not self.default_parameter_train:
            self.__add_default_parameter(self.training_data)
            self.default_parameter_train = True

        average_cost = 0
        for x,y,z,correct_value in data:
            output = self.__sigmoid(np.matmul(self.thetas,np.array([x,y,z])))
            cost = (output - correct_value) ** 2
            average_cost += cost
        
        return average_cost / (len(data) * 2)



    def __sigmoid(self, input):
        return 1.0 / ( 1.0 + np.exp(-input) )
    
    def __add_default_parameter(self, data):
        for item in data:
            item.insert(2,1)

    def __remove_default_parameter(self, data):
        for item in data:
            item.pop(2)

    def predict_test_data(self):
        """ Method used to evaluate the model. """
        if not self.default_parameter_test:
            self.__add_default_parameter(self.test_data)
            self.default_parameter_test = True

        guessed_right = 0
        for point in self.test_data:
            x = point[0]
            y = point[1]
            z = point[2]
            correct_value = point[3]

            prediction = self.__sigmoid(np.matmul(self.thetas,np.array([x,y,z])))
            if prediction < 0.5:
                prediction = 0
            else:
                prediction = 1
            if correct_value == prediction:
                guessed_right +=1
            else:
                print("Inorrect point: ( {} , {} ) - ".format(x,y),end = "")
                print("Correct value: {}. Prediction : {}.".format(correct_value,prediction))

        return "Accuracy on test data: {}% with logistic regression.".format(guessed_right / len(self.test_data) * 100)
    
    def plot_cost_function(self):
        plt.figure()
        plt.plot(self.costs)
        plt.xlabel("Epochs")
        plt.ylabel("Loss Function")
    

class Neural_Network:
    def __init__(self, training_data = None, test_data = None):
        """
        Has some default data to run tests. Custom added data has to be of the same format:
        first 2 entries of a data element are the 2 features, while 3rd entry is the class 
        it belongs to (either 0 or 1).

        """
        self.contor = 1
        self.contor2 = 0
        if training_data is None:
            self.training_data = [
                [1,1,0],[1,2,0],[1,3,0],
                [2,0,0],[3,1,0],[4,2,0],
                [4,3,1],[4,4,1],[5,4,1],
                [5,5,1],[5,6,1],[6,6,1],
                [7,1,1],[1,10,1],[1,8,1],
                [8,8,1],[2,2,0],[0,3,0],
                [0,0,0],[0,5,0],[4,0,0],
                [2,6,1],[3,7,1],[3,5,1],
                [2,4,0],[1,6,1],[3,3,0],
                [0,6,0],[5,0,0],[0,8,1],
                [0,2.8,0],[0.8,3.8,0],[0,2,0],
                [3,9,1],[6,4,1],[3.2,5.9,1],
                [1.5,2,0],[2.8,0,0],[6,1,1],
                [3.5,4,1],[4.5,8,1],[6.5,9.5,1],
                [1.5,0,0],[2.2,0.4,0],[3.5,0.6,0],
                [1,0,0],[2,8,1],[2.5,1.6,0],
                [2.5,7,1],[3.5,5,1],[3.5,8,1],
                [1,5,0],[0.5,0.5,0],[2,2,0],
                [2.5, 1.5, 0],[7,6,1],[5,2,1],
                [0.1,0.1,0],[0.3,0.5,0],[0.5,3.2,0],
                [4,6.5,1],[4.5,9,1],[1.5,9,1],
                [5.5,4,1],[4.5,7,1],[3,8,1],
                [1.5,4,0],[3.2,2,0],[3.5,0,0]]
            
        if test_data is None:
            self.test_data = [
                              [4,6,1],[2,4.5,1],[0,8.5,1],
                              [0.5,1,0],[1.5,1,0],[6,9,1],
                              [6,2,1],[2,6.5,1],[0,6.5,0],
                              [4.5,1.8,0],[2.5,3.8,0],[5.5,6,1],
                              [1,2.2,0],[0.5,4,0],[2.8,7.3,1]]
            
    def plot_training_data(self):
        for point in self.training_data:
            if point[2]:
                plt.plot(point[0], point[1], marker="o", markersize=10, 
                         markerfacecolor="green", markeredgecolor="green")
            else:
                plt.plot(point[0], point[1], marker="x", markersize=10,
                          markerfacecolor="green", markeredgecolor="green")

    def plot_test_data(self):
        for point in self.test_data:
            if point[2]:
                plt.plot(point[0], point[1], marker="o", markersize=10,
                          markerfacecolor="blue", markeredgecolor="blue")
            else:
                plt.plot(point[0], point[1], marker="x", markersize=10,
                          markerfacecolor="blue", markeredgecolor="blue")
            
    def finite_differences_gradient(self, sizes=[2,3,2] ,epochs=10, learning_rate=0.25):
        """Train the network with the finite differences gradient method."""

        self.__format_data(self.training_data)
        self.costs = []

        h = 0.0001 # A small change in the parmatere

        self.w = [np.random.randn(n,m) for n,m in zip(sizes[1:],sizes[:-1])]
        self.b = [np.random.randn(n,1) for n in sizes[1:]]

        self.nabla_w = [np.random.randn(n,m) for n,m in zip(sizes[1:],sizes[:-1])]
        self.nabla_b = [np.random.randn(n,1) for n in sizes[1:]]

        for epoch in range(epochs):
            old_cost = self.__compute_quadratic_cost(self.training_data)

            self.costs.append(old_cost)
            
            for i in range(len(self.w)):
                for j in range(len(self.w[i])):
                    for k in range(len(self.w[i][j])):
                        self.w[i][j][k] += h
                        new_cost = self.__compute_quadratic_cost(self.training_data)
                        self.w[i][j][k] -= h
                        delta_cost = new_cost - old_cost # Corrresponding small change in the loss function
                        self.nabla_w[i][j][k] = self.w[i][j][k] - learning_rate * ((delta_cost) / h)


            for i in range(len(self.b)):
                for j in range(len(self.b[i])):
                    self.b[i][j] += h
                    new_cost = self.__compute_quadratic_cost(self.training_data)
                    self.b[i][j] -= h
                    delta_cost = new_cost - old_cost
                    self.nabla_b[i][j] =  self.b[i][j] - learning_rate * (delta_cost) / h

            self.__update_parameters()
    
    def __update_parameters(self):
        """Updates parameters after a single iteration of finite differences gradient"""
        for i in range(len(self.w)):
                for j in range(len(self.w[i])):
                    for k in range(len(self.w[i][j])):
                        self.w[i][j][k] = self.nabla_w[i][j][k]

        for i in range(len(self.b)):
            for j in range(len(self.b[i])):
                self.b[i][j] = self.nabla_b[i][j]

    def __format_data(self, data):
        for item in data:
            if item[2]:
                item[2] = [0,1]
            else:
                item[2] = [1,0]

    def __compute_quadratic_cost(self, data):
        average_cost = 0 
        for x,y,correct_value in data:
            for i,j in zip(correct_value,self.__feed_forward(x,y)[0]):
                cost = (i-j)**2
                average_cost += cost
        return average_cost / (len(data) * 2)
    
    def __feed_forward(self, x, y):
        a = np.array([[x], [y]])
        for w,b in zip(self.w, self.b):
            a = self.__sigmoid(np.dot(w, a) + b)
        return a   
    
    def __sigmoid(self, input):
        return 1.0 / ( 1.0 + np.exp(-input) )
    
    def predict_test_data(self):
        """ Method used to evaluate the model. """
        self.__format_data(self.test_data)
        
        guessed_right = 0
        for point in self.test_data:
            x = point[0]
            y = point[1]
            correct_value = point[2]

            prediction = self.__feed_forward(x,y)
            if prediction[1] > prediction[0]:
                prediction = [0,1]
            else:
                prediction = [1,0]
            
            if prediction == correct_value:
                guessed_right += 1
            else:
                print("Incorrect point: ( {} , {} ) - ".format(x,y),end = "")
                print("Correct value: {}. Prediction : {}.".format(correct_value,prediction))

        return "Accuracy on test data: {}% with neural network.".format(guessed_right / len(self.test_data) * 100)

    def plot_cost_function(self):
        plt.figure()
        plt.plot(self.costs)
        plt.xlabel("Epochs")
        plt.ylabel("Loss Function")

def main():
    model1 = Logistic_Regression()
    model2 = Neural_Network()
    
    model1.plot_training_data()
    model1.plot_test_data()  
    
    model1.logistic_regression(epochs=300, learning_rate=0.1) # To remove the oscillasions in the cost graph,
    # one can increase the trainig epochs and decrease the learning rate proportionally, at the cost of training time
    print(model1.predict_test_data())
    model1.plot_cost_function()
    
    # model2.finite_differences_gradient(epochs= 5000, learning_rate = 1) 
    # print(model2.predict_test_data())
    # model2.plot_cost_function()
if __name__ == "__main__":
    main()
