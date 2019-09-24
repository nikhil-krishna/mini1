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
    def sigmoid(self, x):
        # The mapping of the function is in [0,1]
        value = np.dot(self.w, x)
        return 1 / (1 + np.exp(-value))


    # Our Error function
    def cross_entropy(self, X, y):  # y=0,1
        # Here is our cost function
        lost = 0
        for i in range(len(X.index)):

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

lr_wine = LogisticRegression(np.zeros((1, 12), float))
wine_df.insert(0, "Constant", 1)

wine_df_copy = wine_df.copy()
wine_df_copy = wine_df_copy.drop(columns=["quality"])

X = wine_df[wine_df.columns[0:12]]
y = wine_df["quality_modified"]
#fit_results = lr_wine.fit(X[0:1200], y[0:1200], learning_rate=0.0001, iteration=20)
prediction = lr_wine.predict(X[1200:1600])
number = lr_wine.evaluate_acc(y[1200:1600], prediction)


def k_fold_CV(data, model, k, learning_rate, iteration):

    all_data = data.iloc[np.random.permutation(len(data))]
    data_split = np.array_split(data, k)
    accuracies = np.ones(k)

    for i, data in enumerate(data_split):

        training_data = pd.concat([all_data, data_split[i], data_split[i]]).drop_duplicates(keep=False)
        model.fit(training_data[training_data.columns[0:12]], np.array(training_data[training_data.columns[12]]),
                  learning_rate=learning_rate, iteration=iteration)
        prediction = model.predict(data_split[i][data_split[i].columns[0:12]])
        accuracies[i] = model.evaluate_acc(data_split[i][data_split[i].columns[12]], prediction)

    return np.mean(accuracies)

print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.001, 10))
