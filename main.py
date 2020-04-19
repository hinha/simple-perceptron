import numpy as np
from random import randrange


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

    def __init__(self, learning_rate=0.1, epochs=50, data_latih=0):
        self.learning_rate = learning_rate
        self.data_latih = data_latih
        self.epochs = epochs
        self.weights = 0
        self.MSE = np.zeros(epochs + 1)
        # self.weights = np.random.rand(inputs) * 2 - 1

    def predict(self, row, weights):
        activation = weights[0]
        for i in range(len(row)-1):
            activation += weights[i + 1] * row[i]
        return 1.0 if activation >= 0.0 else 0.0

    def train_weights(self, train, l_rate, n_epoch):
        weights = [0.0 for i in range(len(train[0]))]
        ltr = 0
        for epoch in range(self.epochs):
            sum_error = 0.0
            for row in train:
                prediction = self.predict(row, weights)
                error = row[-1] - prediction
                sum_error += error**2
                weights[0] = weights[0] + self.learning_rate * error
                for i in range(len(row)-1):
                    weights[i + 1] = weights[i + 1] + \
                        self.learning_rate * error * row[i]
            print('>epoch=%d, lrate=%.3f, error=%.3f' %
                  (self.epochs, self.learning_rate, sum_error))
            self.MSE[ltr] = sum_error / self.data_latih
            ltr += 1
        self.weights = weights
        return weights

    # Split a dataset into k folds
    def cross_validation_split(self, dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    # Perceptron Algorithm With Stochastic Gradient Descent
    def perceptron(self, train, test, l_rate, n_epoch):
        predictions = list()
        weights = self.train_weights(train, l_rate, n_epoch)
        for row in test:
            prediction = self.predict(row, weights)
            predictions.append(prediction)
        return(predictions)

    def evaluate_algorithm(self, dataset, n_folds=4):
        folds = self.cross_validation_split(dataset, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = self.perceptron(
                train_set, test_set, self.learning_rate, self.epochs)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)

        return scores

    # Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

# source: https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
