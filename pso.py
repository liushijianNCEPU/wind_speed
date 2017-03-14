import numpy as np
from GPR import gpr
from pyswarm import pso
import pandas as pd


if __name__ == '__main__':
    # trainNum = 600
    # testNum = 5
    # featureNum = 10

    # x = np.zeros([trainNum, featureNum])
    # y = np.zeros([trainNum, 1])
    # xs = np.zeros([testNum*12, featureNum])
    # ys = np.zeros([testNum*12, 1])

    # speed = pd.read_csv('D:\Study\Project\Python\wind-speed-forecasting_v1\data\G1_speed.csv')
    # for index in range(trainNum):
    #     y[index, 0] = speed.iloc[(index + featureNum), 1]
    #     x[index, :] = speed.iloc[index:(index + featureNum), 1]
    # for index2 in range(testNum*12):
    #     ys[index2, 0] = speed.iloc[(index2 + featureNum + trainNum), 1]
    #     xs[index2, :] = speed.iloc[(index2 + trainNum):(index2 + trainNum + featureNum), 1]
    trainNum = 600
    testNum = 60
    featureNum = 10

    x = np.zeros([trainNum+testNum, featureNum])
    y = np.zeros([trainNum, 1])
    ys = np.zeros([testNum, 1])
    speed = pd.read_csv('D:\Study\Project\Python\wind-speed-forecasting_v1\data\G2_speed.csv')
    for index in range(trainNum):
        y[index, 0] = speed.iloc[(index + featureNum), 1]
    for index2 in range(testNum):
        ys[index2, 0] = speed.iloc[(index2 + featureNum + trainNum), 1]
    for index3 in range(trainNum+testNum):
        x[index3, :] = speed.iloc[index3:(index3 + featureNum), 1]

    args = [x, y, ys]
    lb = [0.01, 0.01, 0.01]
    ub = [500, 500, 500]

    xopt, fopt = pso(gpr, lb, ub, args=args)
    print(xopt)
