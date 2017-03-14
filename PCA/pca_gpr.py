import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pyGPs


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
min_max_scaler1 = preprocessing.MinMaxScaler()
min_max_scaler2 = preprocessing.MinMaxScaler()
min_max_scaler3 = preprocessing.MinMaxScaler()
x = min_max_scaler1.fit_transform(x)
y = min_max_scaler2.fit_transform(y)
ys = min_max_scaler3.fit_transform(ys)

pca = PCA(n_components='mle')
x_pca = pca.fit_transform(x)

model = pyGPs.GPR()
# k1 = pyGPs.cov.RBF(np.log(0.63),np.log(170.88)) + pyGPs.cov.Noise(np.log(40.71))
k1 = pyGPs.cov.RBF(np.log(500),np.log(500)) + pyGPs.cov.Noise(np.log(0.01))
m = pyGPs.mean.Linear( D=x_pca[0:trainNum,:].shape[1] ) + pyGPs.mean.Const()
model.setData(x_pca[0:trainNum,:], y)
model.setPrior(mean=m, kernel=k1)
ymu, ys2, fmu, fs2, lp = model.predict(x_pca[trainNum:trainNum+testNum,:])

ys = min_max_scaler3.inverse_transform(ys)
ymu = min_max_scaler3.inverse_transform(ymu)
mae = np.sum(abs(ymu-ys)) / testNum
rmse = pyGPs.Validation.valid.RMSE(ymu, ys)
mape = np.sum(abs((ymu-ys) / ys)) / testNum
print('MAE:',mae)
print('RMSE:',rmse)
print('MAPE:',mape)
print(x_pca[0,:])