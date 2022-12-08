from MiniProjectPath1 import dataset_1
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression

def input_X_matrix():
    """returns a 5 column matrix such that data for each bridge is recorded in a single colmn vector"""
    Brooklyn, Manhattan = list(dataset_1['Brooklyn Bridge']), list(dataset_1['Manhattan Bridge'])
    Williamsburg, Queensboro = list(dataset_1['Williamsburg Bridge']), list(dataset_1['Queensboro Bridge'])
    return np.array([Brooklyn, Manhattan, Williamsburg, Queensboro], 'F')


def main():

if __name__ == '__main__':
    main()
