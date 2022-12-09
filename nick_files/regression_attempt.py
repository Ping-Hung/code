from MiniProjectPath1 import dataset_1
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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


def regression(X, y):
    norm_X, norm_y = normalize(X), normalize(y)
    return LinearRegression(fit_intercept = True).fit(X,y)

def normalize(vector):
    # vector is a list, shall return normalized vector
    return (vector - np.mean(vector, axis = 0)) / np.std(vector, axis = 0)

def main():
    X = input_X_matrix()
    y = output_y_matrix()


    """five fold CV"""
    # model_and_score = []
    #
    # for idx in range(5):
    #     x_train, x_test, y_train, y_test = split_data(X,y)
    #     model = regression(normalize(x_train), normalize(y_train))
    #     r_square = model.score(x_test, y_test)
    #     model_and_score.append((model, r_square))
    #
    # for model, score in model_and_score:
    #     print(type(model), end = '\t')
    #     print(score)

    """selecting the best model and score"""

    # best_score = 0
    # best_model = None
    #
    # for model, score in model_and_score:
    #     if score > best_score:
    #         best_score = score
    #         best_model = model

    x_train, x_test, y_train, y_test = split_data(X,y)
    best_model = regression(normalize(x_train), normalize(y_train))
    best_r_square = best_model.score(x_test, y_test)

    """this part is just for printing out results"""
    print('equation:')
    for num, x in zip(best_model.coef_, ['Brooklyn', 'Manhattan', 'Williamsburg', 'Queensboro']):
        print(str(num) + x, end = '')
        if x != 'Queensboro':
            print(' + ', end = '')
    print(' + ' + str(best_model.intercept_))

    print(f"\nintercept of the model is {best_model.intercept_}")
    print(f"r^2 value = {best_r_square}")


if __name__ == '__main__':
    # print(normalize(input_X_matrix()))
    main()
