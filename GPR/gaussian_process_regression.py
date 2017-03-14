import pyGPs
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA


def gpr(p, *args):
    p1 = p[0]
    p2 = p[1]
    p3 = p[2]
    # x, y, xs, ys = args
    # min_max_scaler1 = preprocessing.MinMaxScaler()
    # min_max_scaler2 = preprocessing.MinMaxScaler()
    # min_max_scaler3 = preprocessing.MinMaxScaler()
    # min_max_scaler4 = preprocessing.MinMaxScaler()
    # x = min_max_scaler1.fit_transform(x)
    # xs = min_max_scaler2.fit_transform(xs)
    # y = min_max_scaler3.fit_transform(y)
    # ys = min_max_scaler4.fit_transform(ys)

    x, y, ys = args
    min_max_scaler1 = preprocessing.MinMaxScaler()
    min_max_scaler2 = preprocessing.MinMaxScaler()
    min_max_scaler3 = preprocessing.MinMaxScaler()
    x = min_max_scaler1.fit_transform(x)
    y = min_max_scaler2.fit_transform(y)
    ys = min_max_scaler3.fit_transform(ys)

    pca = PCA(n_components='mle')
    x_pca = pca.fit_transform(x)

    k1 = pyGPs.cov.RBF(np.log(p1), np.log(p2)) + pyGPs.cov.Noise(np.log(p3))
    # STANDARD GP (prediction)
    m = pyGPs.mean.Linear( D=x_pca[0:600,:].shape[1] ) + pyGPs.mean.Const()
    model = pyGPs.GPR()
    model.setData(x_pca[0:600,:], y)
    model.setPrior(mean=m, kernel=k1)

    # STANDARD GP (training)
    # model.optimize(x, y)
    ymu, ys2, fmu, fs2, lp = model.predict(x_pca[600:660,:])

    ymu = min_max_scaler3.inverse_transform(ymu)
    ys = min_max_scaler3.inverse_transform(ys)

    rmse = pyGPs.Validation.valid.RMSE(ymu, ys)
    return rmse
