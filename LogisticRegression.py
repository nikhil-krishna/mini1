import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as mp
from itertools import chain, combinations
import time

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
    def sigmoid(self, x):
        # The mapping of the function is in [0,1]
        value = np.dot(self.w, x)

        if value >= 0:
            z = np.exp(-value)
            return 1 / (1+z)
        else:
            z = np.exp(value)
            return z / (1+z)


    # Our Error function
    def cross_entropy(self, X, y):  # y=0,1
        # Here is our cost function
        lost = 0
        for i in range(len(X.index)):

            if self.sigmoid(np.array(X.iloc[i])) == 1 or self.sigmoid(np.array(X.iloc[i])) == 0:
                continue

            lost += -(y[i] * np.log(self.sigmoid(np.array(X.iloc[i]))) + (1 - y[i]) *
                      np.log(1 - self.sigmoid(np.array(X.iloc[i]))))
        return lost/len(X.index)


    # gradient of error function
    def gradient_cross_entropy(self, X, y):
        # Later for the gradient descent
        # Compute gradient at a certain point w
        gradient_at_w = 0
        for i in range(len(X.index)):
            gradient_at_w += - (np.array(X.iloc[i]) * (y[i] - self.sigmoid(np.array(X.iloc[i]))))
        return gradient_at_w


    # Gradient descent, we are updating w, started from randomly generated w0=[0,100]
    # w=w'-(learning rate)*gradient of error function
    # gradient descent step
    def fit(self, X, y, learning_rate=0.001, iteration=50000):
        difference = 1.0
        min_difference = 0.1  # difference between w and w'
        iteration_counter = 0
        cross_en = []

        while iteration_counter <= iteration:
           # and difference >= min_difference:
            gradient = self.gradient_cross_entropy(X, y)
            cost = self.cross_entropy(X, y)
            store_w = self.w

            self.w = store_w - learning_rate * gradient  # making parameters
            cross_en.append(cost)
            #difference = np.sum(list(self.w - store_w))
            iteration_counter = iteration_counter + 1


        return self.w, cross_en


    # predicting the data points question!!!!!!!
    def predict(self, X):
        prediction = []
        for i in range(len(X.index)):
            number = self.sigmoid(np.array(X.iloc[i]))
            if number == 1:
                prediction.append(1)
            elif number == 0:
                prediction.append(0)
            else:
                prediction.append(float(np.log(number/(1-number))))

        prediction = [0 if pred <= 0 else 1 for pred in prediction]

        return prediction


    def evaluate_acc(self, y, prediction_y):

        y = list(y)

        for i in range(len(prediction_y)):

            differences = np.subtract(y, prediction_y)
            return (len(differences) - np.sum(np.abs(differences))) / len(differences)




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

lr_wine = LogisticRegression(np.zeros((1, 13), float))
wine_df.insert(0, "Constant", 1)
#wine_df.insert(12, "Interaction", wine_df["alcohol"]*wine_df["sulphates"])
# Add second interaction term
wine_df.insert(12, "Interaction 2", wine_df["total sulfur dioxide"]*wine_df["sulphates"])

wine_df_copy = wine_df.copy()
wine_df_copy = wine_df_copy.drop(columns=["quality"])

X = wine_df[wine_df.columns[0:13]]
y = wine_df["quality_modified"]

def k_fold_CV(data, model, k, learning_rate, iteration):

    all_data = data.iloc[np.random.permutation(len(data))]
    data_split = np.array_split(data, k)
    accuracies = np.ones(k)

    for i, data in enumerate(data_split):

        training_data = pd.concat([all_data, data_split[i], data_split[i]]).drop_duplicates(keep=False)
        model.fit(training_data[training_data.columns[0:len(data.columns)-1]], np.array(training_data[training_data.columns[len(data.columns)-1]]),
                  learning_rate=learning_rate, iteration=iteration)
        prediction = model.predict(data_split[i][data_split[i].columns[0:len(data.columns)-1]])
        accuracies[i] = model.evaluate_acc(data_split[i][data_split[i].columns[len(data.columns)-1]], prediction)

    return np.mean(accuracies)







# print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.0001, 10)) # 0.7254565047021944



# print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.05, 10)) # 0.646007053291536
# print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.1, 10)) # 0.6466555642633228
# print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.001, 20)) # 0.7254584639498433
# print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.0001, 30)) # 0.7385795454545455

# print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.001, 10)) # 0.7504663009404389 - with second interaction term
# print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.001, 15)) # 0.7517202194357366 - with second interaction term
# print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.001, 25)) # 0.7523491379310345 - with second interaction term
# print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.01, 10)) # 0.6666457680250784 - with second interaction term
# print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.1, 10)) # 0.6504173197492162 - with second interaction term

