# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    x, y = read_data()
    e = np.eye(6)
    alpha = -0.1
    weight = np.matmul(np.linalg.inv(np.matmul(x.T, x) + np.matmul(alpha, e)), np.matmul(x.T, y))
    print(np.matmul(alpha, e))
    return data @ weight
    
def lasso(data):
    x, y = read_data()
    weight = np.array([0, 0, 0, 0, 0, 0])
    label = 2e-5
    alpha = 0.01
    a = 1e-12
    for i in range(int(2e6)):
        z = np.dot(x, weight)
        loss = np.dot((z - y).T, z - y) + alpha * np.sum(abs(weight))
        if loss < label:
            break
        dw = np.dot(z-y, x) + np.sign(weight)
        weight = weight - a * dw
    return data @ weight


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y