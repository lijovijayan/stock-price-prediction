import numpy as np
import matplotlib.pyplot as plt
from store import years, gdp, inflation_rate, unemployment_rate, fiscal_deficit, exchange_rate, stock_price


# plotting the data on a line chart
def plot_data():
    plt.plot(years, gdp, marker='x', c='r')
    plt.plot(years, inflation_rate, marker='o', c='b')
    plt.plot(years, unemployment_rate, marker='o', c='orange')
    plt.plot(years, fiscal_deficit, marker='o', c='y')
    plt.plot(years, exchange_rate, marker='o', c='g')
    plt.title("GDP rate: from 1980 to 2021")
    plt.xlabel("Year")
    plt.ylabel("GDP (in billions USD)")
    plt.show()


def generate_dataset():
    X = []
    for x1_i, x2_i, x3_i, x4_i, x5_i in zip(gdp, inflation_rate, unemployment_rate, fiscal_deficit, exchange_rate):
        training_example = [x1_i, x2_i, x3_i, x4_i, x5_i]
        X.append(training_example)
    return np.array(X)


def plot_cost(cost_list):
    plt.plot(cost_list)
    plt.show()


def predict(X, W, b):
    fx = np.dot(X, W) + b
    return fx


def compute_cost(X, y, W, b):
    m = X.shape[0]

    total_cost = 0
    for xi, yi in zip(X, y):
        total_cost += np.square(predict(xi, W, b) - yi)

    cost = (1 / (2 * m)) * total_cost
    return cost


# calculate the partial derivatives
def dj_dw(X, y, W, b):
    djdw = np.zeros(X.shape[1])
    m = X.shape[0]
    for xi, yi in zip(X, y):
        djdw += np.multiply(xi, (predict(xi, W, b) - yi))
    return (1 / m) * djdw


# calculate the partial derivatives
def dj_db(X, y, W, b):
    djdb = 0
    m = X.shape[0]
    for xi, yi in zip(X, y):
        djdb += (predict(xi, W, b) - yi)
    return (1 / m) * djdb


def compute_gradient(X, y, W, b, alpha):
    temp_w = W - (alpha * dj_dw(X, y, W, b))
    temp_b = b - (alpha * dj_db(X, y, W, b))
    return temp_w, temp_b


def gradient_descent(X, y, n, W, b, alpha):
    cost_list = np.array([])

    for i in range(0, n):
        W, b = compute_gradient(X, y, W, b, alpha)
        cost = compute_cost(X, y, W, b)
        cost_list = np.append(cost_list, cost)

    # print(cost_list)
    # plot_cost(cost_list)
    return W, b


def init():
    X = generate_dataset()
    y = stock_price
    W = np.zeros(X.shape[1])
    b = 0
    alpha = 0.01
    num_iters = 500
    W, b = gradient_descent(X, y, num_iters, W, b, alpha)
    print(f"W: {W}")
    print(f"B: {b}")

    test_data_index = 1
    test_data = X[test_data_index]
    expected_output = y[test_data_index]
    predicted_output = predict(test_data, W, b)

    print(f"expected output: {expected_output} \n predicted output: {predicted_output}")


init()
