import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


trainNum = 600
featureNum = 10
x = np.linspace(1, 6, 6)
y = np.zeros([trainNum, featureNum])
speed = pd.read_csv('D:\Study\Project\Python\wind-speed-forecasting_v1\data\G1_speed.csv')
# for i in range(trainNum):
#     y[i, 0] = speed.iloc[i, 1]
for index3 in range(trainNum):
	y[index3, :] = speed.iloc[index3:(index3 + featureNum), 1]
# c = np.load('D:\Study\Project\Python\wind-speed-forecasting_v1\data\dA\G1_highlevel.npy')
# y[:, 0] = c[0, 0 , :]
pca = PCA(n_components=6)
y_pca = pca.fit_transform(y)
yp = y_pca[100,:].T
plt.figure(figsize=(8,4))
plt.plot(x,yp,color="blue",linewidth=1)
plt.xlabel("Feature number")
plt.ylabel("PCA feature")
# plt.title("PyPlot First Example")
# plt.ylim(-1.2,1.2)
plt.legend()
plt.show()