"""
Linearly separable case SVM, no kernels, and optimization problem is solved by "cvxpy" library
I am minimizing the squared norm2 of the w vector with respect to w and b and subject to the constraints that each
training example must have functional margin >= 1.
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy


class SVM:
    def __init__(self, split = 0.7):
        """Create 2 classes of training data and split them accordingly."""
        samples_per_class0 = 100
        samples_per_class1 = 100

        split = int((samples_per_class0 + samples_per_class1) * split)

        mean_class_0 = np.array([-1.5,-1.5])
        covariance_class_0 = np.array([[0.5, 0.], [0., 0.5]])
        self.features_class_0 = np.random.multivariate_normal(mean_class_0,covariance_class_0,samples_per_class0)

        mean_class_1 = np.array([1.0,2])
        covariance_class_1 = np.array([[0.5, -0.1], [-0.1, 0.5]])
        self.features_class_1 = np.random.multivariate_normal(mean_class_1,covariance_class_1,samples_per_class1)

        features_class_0_zeroed = np.hstack((self.features_class_0, np.zeros((samples_per_class0, 1))))
        features_class_1_zeroed = np.hstack((self.features_class_1, np.ones((samples_per_class1, 1))))

        self.training_data = np.vstack((features_class_0_zeroed,features_class_1_zeroed))
        np.random.shuffle(self.training_data)

        self.test_data = self.training_data[split:]
        self.training_data = self.training_data[:split]

    def plot_data(self):
        """Reformats the traianing data and prints it""" 
        data = self.training_data

        points_with_0 = data[data[:, 2] == 0]
        points_with_1 = data[data[:, 2] == 1]


        plt.scatter(points_with_0[:, 0], points_with_0[:, 1], color='red', label='Class 0')
        plt.scatter(points_with_1[:, 0], points_with_1[:, 1], color='orange', label='Class 1')

        plt.legend()

    def plot_test_data(self):
        """Reformats the testing data and prints it""" 
        data = self.test_data

        points_with_0 = data[data[:, 2] == 0]
        points_with_1 = data[data[:, 2] == 1]


        plt.scatter(points_with_0[:, 0], points_with_0[:, 1], color='red', edgecolors="green")
        plt.scatter(points_with_1[:, 0], points_with_1[:, 1], color='orange', edgecolors="green")
        plt.title("Points circled in green are testing points")

        plt.legend()

    def fit(self):
        """Solves the optimization problem to get the optimal values for w and b"""

        # Reformatting data to a format that is usable by the solver, meaning labels in an array and data in a vector

        len_data, len_features = self.training_data.shape

        y_values = np.delete(self.training_data, 1, axis=1)
        y_values = np.delete(y_values, 0, axis=1)

        for i in range(len_data):
            y_values[i][0] = 1. if y_values[i][0] == 1 else -1.

        X_values = np.delete(self.training_data, 2 , axis= 1)

        len_features -= 1

        # Defining and solving the optimization problem

        w = cvxpy.Variable((len_features, 1))
        b = cvxpy.Variable()

        objective = cvxpy.Minimize(cvxpy.sum_squares(w))

        print("Yvakues",y_values[3][0])
        print("xval",X_values[3].reshape(1,-1).shape)
        print(w)
        constraints = [y_values[i][0] * (X_values[i, :].reshape(1,-1) @ w + b) >= 1 for i in range(len_data)]

        problem = cvxpy.Problem(objective, constraints)

        problem.solve()


        optimal_w = w.value
        self.optimal_w = optimal_w
        optimal_b = b.value
        self.optimal_b = optimal_b


        print("Optimal w:", optimal_w)
        print("Optimal b:", optimal_b)


        # Plotting decision boundary

        x_axis = np.linspace(-3,3,10)
        plt.plot(x_axis, (-optimal_b - optimal_w[0][0]*x_axis) / optimal_w[1][0], linestyle='dashed', label = "Decision boundary")

        # Plotting support vectors

        x_axis = np.linspace(-3,3,10)
        plt.plot(x_axis, (1 -optimal_b - optimal_w[0][0]*x_axis) / optimal_w[1][0], color = "orange", label = "Support vector")

        x_axis = np.linspace(-3,3,10)
        plt.plot(x_axis, (-1 -optimal_b - optimal_w[0][0]*x_axis) / optimal_w[1][0], color = "red" , label = "Support vector")

        plt.legend()

    def run_test(self):
        guessed = 0
        len_test = len(self.test_data)
        for point in self.test_data:
            prediction =  point[:2].reshape(1,-1)  @ self.optimal_w  + self.optimal_b
            prediction = prediction[0][0]
            if prediction >= 0:
                prediction = 1.
            else:
                prediction = 0.

            if prediction == point[2]:
                guessed += 1
        
        print("Accuracy on test data: {}%".format(100*guessed/len_test))
    
if __name__ == "__main__":
    model  = SVM()
    model.plot_data()
    model.plot_test_data()
    model.fit()
    model.run_test()

