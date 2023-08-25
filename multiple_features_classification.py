import numpy as np
import matplotlib.pyplot as plt

class MultipleFeatures:
    def __init__(self, training_data = None, test_data = None):
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
                [2.5,7,1],[3.5,5,1],[3.5,8,1]]
            
        if test_data is None:
            self.test_data = [[1,5,0],[0.5,0.5,0],[2,2,0],
                              [2.5, 1.5, 0],[7,6,1],[5,2,1],
                              [4,6,1],[2,4.5,1],[0,8.5,1],
                              [0.5,1,0],[1.5,1,0],[6,9,1],
                              [6,2,1],[2,6.5,1],[0,6.5,0],
                              [4.5,1.8,0],[2.5,3.8,0]]
            
        self.thetas = np.random.randn(1,3)


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
                          markerfacecolor="green", markeredgecolor="blue")
                
    def logistic_regression(self, epochs = 15, learning_rate = 0.01):

        nabla_thetas = np.zeros((1,3))
        
        for epoch in range(epochs):
            
            nabla_thetas[0][0] = sum([ (prediction - self.sigmoid(np.matmul(self.thetas,np.array([x,y,z]))))*(-x) 
                                for x,y,z,prediction in self.training_data])
            nabla_thetas[0][1] = sum([ (prediction - self.sigmoid(np.matmul(self.thetas,np.array([x,y,z]))))*(-y) 
                                for x,y,z,prediction in self.training_data])
            nabla_thetas[0][2] = sum([ (prediction - self.sigmoid(np.matmul(self.thetas,np.array([x,y,z]))))*(-z) 
                                for x,y,z,prediction in self.training_data])

    
            self.thetas = [t-learning_rate*n_t for t,n_t in zip(self.thetas, nabla_thetas)]

    def sigmoid(self, input):
        return 1.0 / ( 1.0 + np.exp(-input) )
    
    def add_default_parameter(self, data):
        for item in data:
            item.insert(2,1)

    def predict_test_data(self):
        guessed_right = 0
        for point in self.test_data:
            x = point[0]
            y = point[1]
            z = point[2]
            correct_value = point[3]

            prediction = self.sigmoid(np.matmul(self.thetas,np.array([x,y,z])))
            if prediction < 0.5:
                prediction = 0
            else:
                prediction = 1
            if correct_value == prediction:
                guessed_right +=1
            else:
                print("Inorrect point: (",x,",",y,end=" ) - ")
                print("Correct value: {}. Prediction : {}.".format(correct_value,prediction))

        return "Accuracy on test data: "+str(guessed_right/len(self.test_data)*100)+"%"

model = MultipleFeatures()
model.plot_training_data()
model.plot_test_data()  

model.add_default_parameter(model.training_data)
model.add_default_parameter(model.test_data)

model.logistic_regression(epochs=90, learning_rate=0.5)
print(model.predict_test_data())