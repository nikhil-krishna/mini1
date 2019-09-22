import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as mp

# Logistic regression model
# P(y=1|x)=1/(1+e^-a)
# P(y=0|x)=1-1/(1+e^-a)=(e^-a)/(1-e^-a)
# a=ln(P(y=1|x)/P(y=0|x))=ln1-ln(e^-a)

class LogisticRegression:

    def __init__(self, w):

        self.w = w

    # initializing parameter array
    def initialization_w(self, n):
        w = np.zeros((1, n))
        return w


    # predict the target from given A=wTx
    def logistic(self, a):
        # The mapping of the function is in [0,1]
        return 1 / (1 + np.exp(-a))


    # logistic function
    def derivative_logistic(self, a):
        return self.logistic(a) * (1 - self.logistic(a))


    # Computing wTx
    def information_matrix(self, w, x):
        return np.dot(w.T, x)


    # Compute the probability value
    def result_from_logistic(self, w, x):  # probability function
        return self.logistic(self.information_matrix(w, x))


    # Our Error function
    def cross_entropy(self, w, X, y):  # y=0,1
        # Here is our cost function
        lost = 0
        for i in range(len(X.index)):
            lost += - (y[i] * np.log(self.result_from_logistic(w, np.array(X.loc[i]))) + (1 - y[i]) *
                       np.log(1 - self.result_from_logistic(w, np.array(X.loc[i]))))
        return lost


    # gradient of error function
    def gradient_cross_entropy(self, w, X, y):
        # Later for the gradient descent
        # Compute gradient at a certain point w
        gradient_at_w = 0
        for i in range(len(X.index)):
            gradient_at_w += -X.loc[i] * (y[i] - self.result_from_logistic(w, np.array(X.loc[i])))
        return gradient_at_w


    # # Initialization of learning rate alpha
    # def initialization_learning_rate(self):
    #     rate = 0.05
    #     return rate
    #
    #
    # # Initialization of iteration
    # def initialization_iteration(self):
    #     iteration = 50000
    #     return iteration


    # old fit function
    #  def gradient_descent(x, y):
    #    w = np.ones_like(x)  # Initialize w0
    #    a = initialization_learning_rate()
    #
    #    difference = 1.0
    #    min_difference = 0.0001  # difference between w and w'

    #    max_iteration = 50000  # in case of the function does not converge
    #    iteration_counter = 0
    #    gradient = gradient_cross_entropy(w, x, y)
    #
    #    while iteration_counter <= max_iteration and difference >= min_difference:
    #        store_w = w
    #        w = w - a * gradient_cross_entropy(w, x, y)
    #        difference = abs(w - store_w)
    #        iteration_counter = iteration_counter+1
    #        print(iteration_counter, "iteration", " w-updated: ", w)



    # Gradient descent, we are updating w, started from randomly generated w0=[0,100]
    # w=w'-(learning rate)*gradient of error function
    # gradient descent step
    def fit(self, X, y, learning_rate=0.05, iteration=50000):
        difference = 1.0
        min_difference = 0.1  # difference between w and w'
        iteration_counter = 0
        cross_en = []


        while iteration_counter <= iteration and difference >= min_difference:


            gradient = self.gradient_cross_entropy(self.w, X, y)
            cost = self.cross_entropy(self.w, X, y)
            store_w = self.w

            self.w = self.w - learning_rate * gradient  # making parameters
            cross_en.append(cost)
            difference = np.sum(list(self.w - store_w))
            iteration_counter = iteration_counter + 1


        return self.w, cross_en


    # predicting the data points question!!!!!!!
    def predict(self, X):

        prediction = np.zeros(X.shape[0])
        for i in range(len(X.index)):

            prediction[i] = 1 / (1 + np.exp(-np.dot(self.w, np.array(X.iloc[0][1:]))))

        prediction = [0 if pred < 0.5 else 1 for pred in prediction]

        return prediction


    def evaluate_acc(self, y, prediction_y):

        diff = np.array(y - prediction_y)
        diff = list(diff)
        acc = 0
        for d in diff:
            if d != 0:
                acc += 1

        return acc / (len(diff)*1.0)



wine_df = pd.read_csv("winequality-red.csv", delimiter=";")
wine_df["quality_modified"] = pd.to_numeric((wine_df["quality"] > 5) & (wine_df["quality"] < 11)).astype(int)

# Standardize Data
for column in wine_df.columns[0:11]:
    wine_df[column] = (wine_df[column] - wine_df[column].mean()) / wine_df[column].std()

# comparison_df = wine_df.groupby("quality_modified").mean()
# comparison_df.T.plot(kind="bar")
# plt.show()

# scatter_matrix(wine_df, alpha=0.3)
# plt.show()

lr_wine = LogisticRegression(np.random.rand(12))

wine_df.insert(0, "Constant", 1)
X = wine_df[wine_df.columns[0:12]]
y = wine_df["quality_modified"]
lr_wine.fit(X[0:1200], y[0:1200], learning_rate=0.05, iteration=50000)
prediction = lr_wine.predict(X[1200:1600].reset_index())
print(lr_wine.evaluate_acc(y[1200:1600], prediction))





