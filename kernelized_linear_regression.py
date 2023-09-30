"""
Script that implements kernelized linear regression. Meant to use for experimentation, the  "main" part provides
different tests with different kernel functions. Custom added kernel functions can be used, as long as they folow the format:
they must have 2 parameters, x and x, being column vectors, and returna single value, which is the kernel function evaluated
for the 2 vectors.
"""

import numpy as np
import matplotlib.pyplot as plt

class NotSymmetricMatrix(BaseException):
    pass

class NotPositiveSemidefiniteMatrix(BaseException):
     pass

class Kernelized_Linear_Regression:
    def __init__(self, training_data = None, kernel_function = None):

        if training_data is None:
            self.training_data = self.generate_data()
        else:
            self.training_data = training_data

        self.len_train_data = len(self.training_data)


        if kernel_function is None:
            self.kernel_function = self.default_kernel_function
        else:
            self.kernel_function = kernel_function
            
            
    def generate_data(self):
        """Generates a simple data set for training."""
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

    def plot_points(self, points = None):
        if points is None:
                points = self.training_data
        plt.figure()
        for x,y in points:
            plt.plot(x,y,marker="o",markeredgecolor="green", markerfacecolor="green")
        plt.xlabel("Data")
        plt.ylabel("Price")

    def default_kernel_function(self, x, z):
        """Simple kernel function, equivalent to the extended features containing all the monomials of the base input with degree ≤ 3"""
        xz_dot = np.inner(x,z)
        return 1.0 + xz_dot + xz_dot**2 + xz_dot**3
    
    def compute_kernel_matrix(self):
        len_train_data = len(self.training_data)
        self.kernel_matrix = np.zeros((len_train_data,len_train_data))

        for i in range(len(self.training_data)):
            for j in range(len(self.training_data)):
                self.kernel_matrix[i][j] = self.kernel_function(self.training_data[i][0],self.training_data[j][0])

        if not (self.kernel_matrix.transpose() == self.kernel_matrix).all():
            raise NotSymmetricMatrix("Can't compute kernel matrix: Kernel matrix is not symmetric.")
        
        eigenvalues = np.linalg.eigvals(self.kernel_matrix)
        if not np.all(eigenvalues >= -1e-8):
             raise NotPositiveSemidefiniteMatrix("Can't compute kernel matrix: Kernel matrix is not positive semidefinite.")

    def learn_parameters(self, learning_rate = 0.01, epochs = 10):
        len_train_data = len(self.training_data)
        self.beta = np.zeros((len_train_data,1))

        y = np.zeros((len_train_data,1))
        for i in range(len_train_data):
             y[i][0] = self.training_data[i][1]

        for epoch in range(epochs):
            self.beta = self.beta + learning_rate * (y - self.kernel_matrix @ self.beta)

    def plot_learned_curve(self, title):
        x_axis = np.linspace(-1.5,1.5,100)
        y_axis = self.predict_points(x_axis)
        plt.plot(x_axis,y_axis)
        plt.title(title)

    def predict_points(self,x):
        #return sum(self.beta[i][0]*self.kernel_function(self.training_data[i][0],x) for i in range(self.len_train_data))
        return [sum(self.beta[i][0] * self.kernel_function(self.training_data[i][0], xi) for i in range(self.len_train_data)) for xi in x]
    


# Some kernel functions. Note that for each of these, other hyperparameters have to be chosen.
def square_kernel(x,z):
    xz_dot = np.inner(x,z)
    return (xz_dot + 1/2 )**2

def cube_kernel(x,z):
    xz_dot = np.inner(x,z)
    return (xz_dot + 1/2 )**3       

def high_order_kernel(x,z):
    xz_dot = np.inner(x,z)
    return (xz_dot + 1/2 )**13

def gaussian_kernel_low_sigma(x,z):
    sigma = 0.1
    diff = x - z
    norm = np.linalg.norm(diff)
    return np.exp(-(norm)**2 / (2 * sigma ** 2))

def gaussian_kernel_medium_sigma(x,z):
    sigma = 0.5
    diff = x - z
    norm = np.linalg.norm(diff)
    return np.exp(-(norm)**2 / (2 * sigma ** 2))

def gaussian_kernel_high_sigma(x,z):
    sigma = 1
    diff = x - z
    norm = np.linalg.norm(diff)
    return np.exp(-(norm)**2 / (2 * sigma ** 2))

if __name__ == "__main__":

    model = Kernelized_Linear_Regression()
    model.plot_points()
    model.compute_kernel_matrix()
    model.learn_parameters(learning_rate = 0.01)
    model.plot_learned_curve("Default kernel")



    model = Kernelized_Linear_Regression(kernel_function = square_kernel)
    model.plot_points()
    model.compute_kernel_matrix()
    model.learn_parameters(learning_rate = 0.01)
    model.plot_learned_curve("2-nd order degree kernel")



    model = Kernelized_Linear_Regression(kernel_function = cube_kernel)
    model.plot_points()
    model.compute_kernel_matrix()
    model.learn_parameters(learning_rate = 0.01, epochs = 100)
    model.plot_learned_curve("3-rd order degree kernel")



    try: # This one will take some time, as I had to lower the learning rate a lot, since the model is prone to overshooting while
        # dealing with large values 
        model = Kernelized_Linear_Regression(kernel_function = high_order_kernel)
        model.plot_points()
        model.compute_kernel_matrix()
        model.learn_parameters(learning_rate = 0.000005, epochs = 50000)
        model.plot_learned_curve("High order kernel")
    except NotSymmetricMatrix as e:
        print(e)
    except NotPositiveSemidefiniteMatrix as e:
        print(e)
    except:
        print("Too much!") # Too high values lead to numerical instability in the matrices


    model = Kernelized_Linear_Regression(kernel_function = gaussian_kernel_low_sigma)
    model.plot_points()
    model.compute_kernel_matrix()
    model.learn_parameters(learning_rate = 0.05, epochs = 300)
    model.plot_learned_curve("Gaussian kernel, simga = 0.1")

    model = Kernelized_Linear_Regression(kernel_function = gaussian_kernel_medium_sigma)
    model.plot_points()
    model.compute_kernel_matrix()
    model.learn_parameters(learning_rate = 0.05, epochs = 300)
    model.plot_learned_curve("Gaussian kernel, simga = 0.5")


    model = Kernelized_Linear_Regression(kernel_function = gaussian_kernel_high_sigma)
    model.plot_points()
    model.compute_kernel_matrix()
    model.learn_parameters(learning_rate = 0.005, epochs = 300) # Had to lower learning rate to prevent over-shooting
    model.plot_learned_curve("Gaussian kernel, sigma = 1")


