import numpy as np
import pyGPs
import pandas as pd
from sklearn import preprocessing


trainNum = 600
testNum = 5
featureNum = 10

x = np.zeros([trainNum, featureNum])
y = np.zeros([trainNum, 1])
xs = np.zeros([testNum*12, featureNum])
ys = np.zeros([testNum*12, 1])
prediction = []
real = []

speed = pd.read_csv('D:\Study\Project\Python\wind-speed-forecasting_v1\data\G4_speed.csv')
for index in range(trainNum):
    y[index, 0] = speed.iloc[(index + featureNum), 1]
    x[index, :] = speed.iloc[index:(index + featureNum), 1]
for index2 in range(testNum*12):
	ys[index2, 0] = speed.iloc[(index2 + featureNum + trainNum), 1]
	xs[index2, :] = speed.iloc[(index2 + trainNum):(index2 + trainNum + featureNum), 1]
min_max_scaler1 = preprocessing.MinMaxScaler()
min_max_scaler2 = preprocessing.MinMaxScaler()
min_max_scaler3 = preprocessing.MinMaxScaler()
min_max_scaler4 = preprocessing.MinMaxScaler()
x = min_max_scaler1.fit_transform(x)
xs = min_max_scaler2.fit_transform(xs)
y = min_max_scaler3.fit_transform(y)
ys = min_max_scaler4.fit_transform(ys)

k1 = pyGPs.cov.RBF(np.log(185.03),np.log(137.73)) + pyGPs.cov.Noise(np.log(1.62))
# k2 = pyGPs.cov.RQ()
# k = k1+k2
m = pyGPs.mean.Linear( D=x.shape[1] ) + pyGPs.mean.Const()
model = pyGPs.GPR()
model.setData(x, y)
model.setPrior(mean=m, kernel=k1)
#model.optimize()
ymu, ys2, fmu, fs2, lp = model.predict(xs)
ymu = min_max_scaler4.inverse_transform(ymu)
ys = min_max_scaler4.inverse_transform(ys)
prediction = map(float, ymu)
real = map(float, ys)
data = pd.DataFrame({'prediction':prediction, 'real':real})
data.to_csv('D:\Study\Project\Python\wind-speed-forecasting_v1\data\prediction4.csv')
mae = np.sum(abs(ymu-ys)) / (testNum*12)
rmse = pyGPs.Validation.valid.RMSE(ymu, ys)
mape = np.sum(abs((ymu-ys) / ys)) / (testNum*12)
print('MAE:',mae)
print('RMSE:',rmse)
print('MAPE:',mape)

# for i in range(12):

# 	model = pyGPs.GPR()
# 	model.setData(x, y)
# 	model.optimize()
# 	ymu, ys2, fmu, fs2, lp = model.predict(xs[i*testNum:(i+1)*testNum, :])

# 	ymu = min_max_scaler4.inverse_transform(ymu)
# 	ys1 = min_max_scaler4.inverse_transform(ys)
# 	prediction = prediction + map(float, ymu)
# 	real = real + map(float, ys1[i*testNum:(i+1)*testNum, 0])
# 	# x = np.concatenate((x,xs[i*testNum:(i+1)*testNum, :]))
# 	# y = np.concatenate((y,ys[i*testNum:(i+1)*testNum, :]))
# 	x[i*testNum:(i+1)*testNum, :] = xs[i*testNum:(i+1)*testNum, :]
# 	y[i*testNum:(i+1)*testNum, 0] = ys[i*testNum:(i+1)*testNum, 0]

# # ys = min_max_scaler4.inverse_transform(ys)
# data = pd.DataFrame({'prediction':prediction, 'real':real})
# data.to_csv('D:\Study\Project\Python\wind-speed-forecasting_v1\data\prediction.csv')

# prediction = np.array(prediction)
# real = np.array(real)
# mae = np.sum(abs(prediction-real)) / (testNum*12)
# rmse = pyGPs.Validation.valid.RMSE(prediction, real)
# mape = np.sum(abs((prediction-real) / real)) / (testNum*12)
# print('MAE:',mae)
# print('RMSE:',rmse)
# print('MAPE:',mape)