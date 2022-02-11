# -*- coding: utf-8 -*-
"""Perceptron.ipynb
## Coding up a perceptron
"""

# Commented out IPython magic to ensure Python compatibility.
# Importing Libraries
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

class Perceptron:
    def __init__(self, epochs=3):
        self.epochs = epochs
        self.misclassifications = 0
        self.weights = np.random.rand(1,2)
        self.weights = self.weights * 0
        self.learning_rate = 0.075

    def calculate_error(self, true_label, prediction):
      #Mean Square Error
      return 0.5 * ((true_label - prediction)**2)

    def derivative_error_wrt_output(self, true_label, prediction):
      return -1*(true_label-prediction)

    def derivative_dotP_wrt_weights(self, feature):
      return feature[0:2]

    def sigmoid(self, input):
      return 1 / (1 + math.exp(-input))

    def tanh(self, input):
      return (math.exp(input)-math.exp(-input))/(math.exp(input)+math.exp(-input))

    def derivative_tanh(self, output):
      return 1-output**2

    def derivative_sigmoid(self, output):
      return output * (1-output)

    def calculate_deltas(self, feature, prediction, true_label):
      #2 calculate error
      error = self.calculate_error(feature[-1], prediction)
      #3 derivative of error WRT output
      error_WRT_output = self.derivative_error_wrt_output(true_label, prediction)
      #4 derivative of activation function
      activation_derivative = self.derivative_tanh(prediction)
      #5 derivative of dot product WRT each weight
      gradients = self.derivative_dotP_wrt_weights(feature)
      #6 put everything together to get your deltas
      deltas = error_WRT_output * activation_derivative * self.learning_rate * gradients

      print(deltas.shape)
      return deltas

    def update_weights(self, deltas):
        self.weights = np.subtract(self.weights, deltas)

    def train(self, features, true_labels, plotting=True):
        """
        features: dependent variables (x)
        true_labels: target variables (y)
        plotting: plot the decision boundary (True by default)
        """
        #main training loop
        for epoch in range(self.epochs):
            # Iterate over the training data
            for i in range(len(features)):
                if plotting:
                    print("Iteration {}, Misclassifications = {}".
                      format(epoch * len(features) + i+1, self.misclassifications))
                    self.plot_classifier(features, true_labels, features[i])
                #get the true label
                true_label = true_labels[i]
                #1 forward pass
                prediction = self.predict(features[i][0:2])
                print("Prediction: ", prediction)
                print("True Label: ", true_label)
                if round(prediction) != true_label:
                  self.misclassifications = self.misclassifications + 1
                #2 calculate deltas
                deltas = self.calculate_deltas(features[i], prediction, true_label)
                print("Weights: ", self.weights)
                #3 update weights
                self.update_weights(deltas)

            print("="*25)
            print("Epoch {}, Accuracy = {}".format(epoch + 1, 1 - self.misclassifications/len(features)))
            print("="*25)
            self.misclassifications = 0

    def predict(self, features):
        """
        Predict the label using self.weights.

        Args:
            features: dependent variables (x)

        Returns:
            The predicted label.
        """
        #1 multiply weights vector by the input vector
        input_dot_weights = np.dot(self.weights, features)
        #2 pass through tanh activation function
        output = self.tanh(input_dot_weights)
        #3 return output
        return output

    def plot_classifier(self, features, true_labels, data_point):
        """
        Plot the decision boundary.

        Args:
            features: dependent variables (x)
            true_labels: target variables (y)
            data_point: the current data point under consideration
        """
        # Create a mesh to plot
        x1_min, x1_max = features[:, 0].min() - 2, features[:, 0].max() + 2
        x2_min, x2_max = features[:, 1].min() - 2, features[:, 1].max() + 2
        x1x1, x2x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                             np.arange(x2_min, x2_max, 0.02))

        Z = np.zeros(x1x1.shape)
        fig, ax = plt.subplots()
        for i in range(len(x1x1)):
            for j in range(len(x1x1[0])):
                Z[i,j] = self.predict([x1x1[i,j], x2x2[i,j]])

        # Put the result into a color plot
        ax.contourf(x1x1, x2x2, Z, cmap='bwr', alpha=0.3)

        # Plot the training points
        plt.scatter(np.reshape(features[:, 0], (10,1)), features[:, 1], c=true_labels, cmap='bwr')
        plt.plot(data_point[0], data_point[1], color='k', marker='x', markersize=12)

        ax.set_title('Perceptron')

        plt.show()

"""**Below is a linearly separable toy dataset.**"""

data = np.array([[2.7810836,2.550537003,-1],
        [1.465489372,2.362125076,-1],
        [3.396561688,4.400293529,-1],
        [1.38807019,1.850220317,-1],
        [3.06407232,3.005305973,-1],
        [7.627531214,2.759262235,1],
        [5.332441248,2.088626775,1],
        [6.922596716,1.77106367,1],
        [8.675418651,-0.242068655,1],
        [7.673756466,3.508563011,1]])

"""Normalize and shuffle the dataset"""
#norm = np.max(data)
#data[:, [0,1]] = data[:, [0,1]]/norm
#np.random.shuffle(data)
print(data)
features = data[:, [0, 1]]
true_labels = data[:, [2]]

plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=data[:, -1], cmap='bwr')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

"""Train Perceptron"""
perceptron = Perceptron()
perceptron.train(features, true_labels)

for feature in features:
  out = perceptron.predict(feature)
  print(round(out))
