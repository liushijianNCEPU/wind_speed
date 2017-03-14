import numpy as np
import pandas as pd
from sklearn import preprocessing
import pyGPs


trainNum = 600
testNum = 60
featureNum = 6

x = np.zeros([trainNum, featureNum])
y = np.zeros([trainNum, 1])
xs = np.zeros([testNum, featureNum])
ys = np.zeros([testNum, 1])
c = np.load('D:\Study\Project\Python\wind-speed-forecasting_v1\data\dA\G4_highlevel.npy')
for i in range(trainNum/10):
	for j in range(10):
		x[10*i+j, :] = c[i, j , :]
for i in range(testNum/10):
	for j in range(10):
		xs[10*i+j, :] = c[i+trainNum/10, j , :]
speed = pd.read_csv('D:\Study\Project\Python\wind-speed-forecasting_v1\data\G4_speed.csv')
for index in range(trainNum):
	y[index, 0] = speed.iloc[(index + featureNum), 1]
for index2 in range(testNum):
	ys[index2, 0] = speed.iloc[(index2 + featureNum + trainNum), 1]
min_max_scaler1 = preprocessing.MinMaxScaler()
min_max_scaler2 = preprocessing.MinMaxScaler()
y = min_max_scaler1.fit_transform(y)
ys = min_max_scaler2.fit_transform(ys)

model = pyGPs.GPR()
model.setData(x, y)
model.optimize()
ymu, ys2, fmu, fs2, lp = model.predict(xs)

ys = min_max_scaler2.inverse_transform(ys)
ymu = min_max_scaler2.inverse_transform(ymu)
mae = np.sum(abs(ymu-ys)) / testNum
rmse = pyGPs.Validation.valid.RMSE(ymu, ys)
mape = np.sum(abs((ymu-ys) / ys)) / testNum
print('MAE:',mae)
print('RMSE:',rmse)
print('MAPE:',mape)