import numpy as np

from MiniProjectPath1 import dataset_1

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

def input_X_matrix():
    """returns a 5 column matrix such that data for each bridge is recorded in a single colmn vector"""
    Brooklyn, Manhattan = list(dataset_1['Brooklyn Bridge']), list(dataset_1['Manhattan Bridge'])
    Williamsburg, Queensboro = list(dataset_1['Williamsburg Bridge']), list(dataset_1['Queensboro Bridge'])
    return np.array([Brooklyn, Manhattan, Williamsburg, Queensboro]).transpose()

def output_y_matrix():
    return np.array(list(dataset_1['Total'])).transpose()

def split_data(X, y):
    """shall return 4 lists of data: x_train, x_test; y_train, y_test"""
    #using a 80-20 split: temporarily use None for random_state
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= None)
    return x_train, x_test, y_train, y_test

def knn(X, y, k):
    return KNeighborsClassifier(n_neighbors=k).fit(X,y)

def normalize(vector):
    # vector is a list, shall return normalized vector
    return (vector - np.mean(vector, axis = 0)) / np.std(vector, axis = 0)

if __name__ == '__main__':
    # print(normalize(input_X_matrix()))
    X = input_X_matrix()
    y = output_y_matrix()
    x_train, x_test, y_train, y_test = split_data(X,y)

    k = 4
    knn_model = knn(x_train, y_train, k)

    response = knn_model.predict(x_test)

    acc = metrics.accuracy_score(y_test, response)
    print(acc)

    print(response)