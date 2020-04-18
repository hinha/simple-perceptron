import numpy as np


class Perceptron(object):

    def __init__(self, inputs, threshold=10, learning_rate=0.8):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(inputs + 1)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            print("literate: ", _)
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * \
                    (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
        print(self.weights)


class Perceptron2(object):

    def __init__(self, inputs, learning_rate=0.8, epochs=50, length_data=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(inputs)

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs, weights):
        activation = 0.0
        # print(inputs)
        for i, w in zip(inputs, weights):
            activation += i*w
        return 1.0 if activation >= 0.0 else 0.0

    def accuracy(self, inputs, weights):
        num_correct = 0.0
        preds = []
        for i in range(len(inputs)):
            pred = self.predict(inputs[i][:-1], weights)
            preds.append(pred)
            if pred == inputs[i][-1]:
                num_correct += 1.0
        print("Predictions:", preds)
        return num_correct/float(len(inputs))

    def train_w(self, training_inputs, stop_early=True):
        for _ in range(self.epochs):
            print("ltr: {}".format(_))
            cur_acc = self.accuracy(training_inputs, self.weights)
            print("\nEpoch %d \nWeights: " % self.epochs, self.weights)
            print("Accuracy: ", cur_acc)

            if cur_acc == 1.0 and stop_early:
                break

            for i in range(len(training_inputs)):
                prediction = self.predict(
                    training_inputs[i][:-1], self.weights)

                error = training_inputs[i][-1]-prediction
                for j in range(len(self.weights)):
                    self.weights[j] = self.weights[j] + \
                        (self.learning_rate*error*training_inputs[i][j])
