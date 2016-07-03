import numpy as np
import pyov2sgd
import time
import operalib
import matplotlib.pyplot as plt
import scipy.io as sio

from sklearn.cross_validation import KFold
from sklearn.utils import shuffle
from sklearn import preprocessing

from scipy.spatial.distance import pdist, squareform
from scipy.optimize import fmin_l_bfgs_b

# np.set_printoptions(threshold=np.nan)
# np.random.seed(np.random.randint(1000))
np.random.seed(0)

data = sio.loadmat('example/sarcos_inv.mat')
X_train = data['sarcos_inv'][:, :21].astype(np.float)
y_train = data['sarcos_inv'][:, 21:].astype(np.float)


data = sio.loadmat('example/sarcos_inv_test.mat')
X_test = data['sarcos_inv_test'][:, :21].astype(np.float)
y_test = data['sarcos_inv_test'][:, 21:].astype(np.float)

scaler_X = preprocessing.StandardScaler().fit(X_train)
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = preprocessing.StandardScaler().fit(y_train)
y_train = scaler_y.transform(y_train)
y_test = scaler_y.transform(y_test)

# print np.cov(y_train.T)
sim = pdist(y_train.T) ** 2
# print sim
M = np.exp(- 1. / np.median(sim) * squareform(sim))
D = np.sum(M, axis=0) + np.diag(M)
L = np.linalg.pinv(np.diag(D) - M)

# print('sim', L)


# print('sigma', sigma)

d = X_train.shape[1]
p = y_train.shape[1]


samples_train = 50
samples_test = y_test.shape[0]
tasks = 1
n = samples_train * tasks

y_train[:, [0, 1, 2, 3, 4, 5, 6]] = y_train[:, [0, 1, 2, 3, 4, 5, 6]]
y_test[:, [0, 1, 2, 3, 4, 5, 6]] = y_test[:, [0, 1, 2, 3, 4, 5, 6]]

lbda = .0000 / n
batch = 50
block = 25
T = 50000 * samples_train / batch
eta0 = .01

pts = 21
bts = 1
nMSE = np.zeros((pts, bts))


def plain2mt(X, y, tasks, samples):
    task_idx = np.empty((tasks, samples), dtype=np.int)
    X_mt = np.empty((samples * tasks, X.shape[1]))
    y_mt = np.empty((samples * tasks, 2))
    for t in range(tasks):
        task_idx[t, :] = np.random.choice(X.shape[0], samples, replace=False)
        X_mt[samples * t:samples * (t + 1), :] = X[task_idx[t, :], :]
        y_mt[samples * t:samples * (t + 1), 1] = np.full(samples, t,
                                                         dtype=np.float)
        y_mt[samples * t:samples * (t + 1), 0] = y_train[task_idx[t, :], t]
    X_mt, y_mt = shuffle(X_mt, y_mt, random_state=0)
    return X_mt, y_mt

def mt2plain(y_mt):
    u, idx = np.unique(y_mt[:, 1], return_inverse=True)
    tasks = len(u)
    y_plain = np.empty((y_mt.shape[0], tasks))
    y_plain.fill(np.NaN)
    for t in range(tasks):
        y_plain[idx == t, t] = y_mt[idx == t, 0]
    return y_plain


for (i, alpha) in enumerate(np.linspace(.1, 1, pts)):
    for j in range(bts):
        X_train_mt, y_train_mt = plain2mt(X_train, y_train,
                                          tasks, samples_train)

        sigma = np.median(pdist(X_train_mt))


        def phi(X, D):
            np.random.seed(0)
            W = 1. / (34 * sigma) * np.random.randn(X.shape[1], D)
            Z = np.dot(X, W)
            return np.hstack((np.cos(Z), np.sin(Z)))

        phi_x = phi(X_train_mt, 25)
        wop = np.dot(np.linalg.pinv(phi_x), y_train_mt[:, 0:1])

        pred_train = np.dot(phi(X_train_mt, 25), wop)
        pred_test = np.dot(phi(X_test, 25), wop)

        score_train = np.nanmean((pred_train - y_train_mt[:, 0:1]) ** 2,
                                 axis=0)
        # score_train = np.nanmean((pred_train - y_train[:, 0:tasks]) ** 2,
        #                          axis=0)
        score_test = np.mean((pred_test - y_test[:, 0:tasks]) ** 2,
                             axis=0)
        # print(1, "T pred: ", time.time() - start)
        # nMSE[i, j] = np.mean(score_test)
        # print score_train, score_test

        for i in range(len(score_test)):
            nMSE_train = score_train
            nMSE_test = score_test
            print("Dimension %d: nMSE = %f%% (training) / %f%% (validation)"
                  % (i + 1, score_train[i] * 100, score_test[i] * 100))

        dim = 0
        n_samples = y_train.shape[0]
        plt.subplot(2, 1, 1)
        plt.plot(y_train[:, 0:1], label="Actual")
        # plt.plot(estimator.predict_dense(X_train)[:, dim],
        #          label="Predicted")

        plt.legend(loc="best")
        plt.title("Output of %d samples from dimension %d (validation set)"
                  % (n_samples, dim + 1))

        plt.subplot(2, 1, 2)
        plt.plot(y_train_mt[:, 0:1], label="Actual")
        plt.plot(pred_train[:, dim],
                 label="Predicted")

        plt.show()

# print nMSE
# print nMSE.mean(axis=1)

# for i in range(len(score_test)):
#     nMSE_train = score_train
#     nMSE_test = score_test
#     print("Dimension %d: nMSE = %f%% (training) / %f%% (validation)"
#           % (i + 1, score_train[i] * 100, score_test[i] * 100))
# try:
#     import pylab
# except:
#     print("Cannot plot the result. Matplotlib is not available.")
#     exit(1)

# dim = 5
# n_samples = y_train.shape[0]
# pylab.plot(y_train[:n_samples, dim], label="Actual")
# pylab.plot(estimator.predict_dense(X_train[:n_samples])[:, dim],
#            label="Predicted")
# pylab.legend(loc="best")
# pylab.title("Output of %d samples from dimension %d (validation set)"
#             % (n_samples, dim + 1))
# pylab.show()

