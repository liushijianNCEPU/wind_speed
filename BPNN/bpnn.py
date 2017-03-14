from pybrain.structure import *
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import numpy as np
import pyGPs
import pandas as pd
from sklearn import preprocessing


trainNum = 600
testNum = 60
featureNum = 10

x = np.zeros([trainNum, featureNum])
y = np.zeros([trainNum, 1])
xs = np.zeros([testNum, featureNum])
ys = np.zeros([testNum, 1])

# x1 = []
# x2 = []
# x3 = []
# x4 = []
# x5 = []
# x6 = []
# x7 = []
# x8 = []
# x9 = []
# x10 = []
# y = []
speed = pd.read_csv('D:\Study\Project\Python\wind-speed-forecasting_v1\data\G4_speed.csv')
for index in range(trainNum):
    y[index, 0] = speed.iloc[(index + featureNum), 1]
    x[index, :] = speed.iloc[index:(index + featureNum), 1]
for index2 in range(testNum):
	ys[index2, 0] = speed.iloc[(index2 + featureNum + trainNum), 1]
	xs[index2, :] = speed.iloc[(index2 + trainNum):(index2 + trainNum + featureNum), 1]
# for index in range(trainNum+testNum):
#     y.append(speed.iloc[(index + featureNum), 1])
#     x1.append(speed.iloc[index, 1])
#     x2.append(speed.iloc[index+1, 1])
#     x3.append(speed.iloc[index+2, 1])
#     x4.append(speed.iloc[index+3, 1])
#     x5.append(speed.iloc[index+4, 1])
#     x6.append(speed.iloc[index+5, 1])
#     x7.append(speed.iloc[index+6, 1])
#     x8.append(speed.iloc[index+7, 1])
#     x9.append(speed.iloc[index+8, 1])
#     x10.append(speed.iloc[index+9, 1])

# data = pd.DataFrame({'x1':x1, 'x2':x2,'x3':x3,'x4':x4,'x5':x5,'x6':x6,'x7':x7,'x8':x8,'x9':x9,'x10':x10,'y':y})
# data.to_csv('D:\Study\Project\Python\wind-speed-forecasting_v1\data\laod.csv')
min_max_scaler1 = preprocessing.MinMaxScaler()
min_max_scaler2 = preprocessing.MinMaxScaler()
min_max_scaler3 = preprocessing.MinMaxScaler()
min_max_scaler4 = preprocessing.MinMaxScaler()
x = min_max_scaler1.fit_transform(x)
xs = min_max_scaler2.fit_transform(xs)
y = min_max_scaler3.fit_transform(y)
ys = min_max_scaler4.fit_transform(ys)

DS = SupervisedDataSet(10,1)
for index3 in range(trainNum):
	DS.addSample(x[index3, :], y[index3, 0])

fnn = FeedForwardNetwork()

inLayer = LinearLayer(10, name='inLayer')
hiddenLayer = SigmoidLayer(7, name='hiddenLayer0')
outLayer = LinearLayer(1, name='outLayer')

fnn.addInputModule(inLayer)
fnn.addModule(hiddenLayer)
fnn.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

fnn.addConnection(in_to_hidden)
fnn.addConnection(hidden_to_out)

fnn.sortModules()

prediction = []
trainer = BackpropTrainer(fnn, DS, verbose = False, learningrate=0.01)
trainer.trainUntilConvergence(maxEpochs=1000)
for index4 in range(testNum):
	prediction.append(fnn.activate(xs[index4, :]))
# prediction = []
# for i in range(12):
# 	trainer = BackpropTrainer(fnn, DS, verbose = False, learningrate=0.01)
# 	trainer.trainUntilConvergence(maxEpochs=1000)
# 	for index4 in range(testNum):
# 		prediction.append(fnn.activate(xs[index4 + i*testNum, :]))
# 		DS.addSample(xs[index4 + i*testNum, :], ys[index4 + i*testNum, 0])

real = min_max_scaler4.inverse_transform(ys)
prediction = min_max_scaler4.inverse_transform(prediction)
prediction = np.array(prediction)
mae = np.sum(abs(prediction-real)) / testNum
rmse = pyGPs.Validation.valid.RMSE(prediction, real)
mape = np.sum(abs((prediction-real) / real)) / testNum
print('MAE:',mae)
print('RMSE:',rmse)
print('MAPE:',mape)