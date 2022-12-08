from MiniProjectPath1 import dataset_1
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def input_X_matrix(test_indeces):
    """returns a 5 column matrix such that data for each bridge is recorded in a single colmn vector"""
    Brooklyn, Manhattan = list(dataset_1['Brooklyn Bridge']), list(dataset_1['Manhattan Bridge'])
    Williamsburg, Queensboro = list(dataset_1['Williamsburg Bridge']), list(dataset_1['Queensboro Bridge'])
    return np.matrix([Brooklyn, Manhattan, Williamsburg, Queensboro]).transpose()

def output_y_matrix():
    return np.matrix(list(dataset_1['Total'])).transpose()

def split_data(X, y):
    """shall return 4 lists of data: x_train, x_test; y_train, y_test"""
    #using a 80-20 split: temporarily use None for random_state
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state= None)
    return x_train, x_test, y_train, y_test

def regression(X, y):


def main():

if __name__ == '__main__':
    main()
