from pyov2sgd import *
import numpy as np
import time
from scipy.spatial.distance import pdist
from sklearn import decomposition
# from sklearn.cross_validation import train_test_split
# from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import LabelBinarizer

# Generate data
from sklearn.datasets import fetch_mldata
np.random.seed(0)
mnist = fetch_mldata('MNIST original')

X, y = mnist.data, mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)

lb = LabelBinarizer(neg_label=0, pos_label=1)
y_train = lb.fit_transform(y_train).astype(np.float64)
y_test = lb.transform(y_test).astype(np.float64)

ppross = decomposition.RandomizedPCA(n_components=50, whiten=True)
ppross.fit(X_train)
X_train = ppross.transform(X_train)
X_test = ppross.transform(X_test)

n = X_train.shape[0]
p = y_train.shape[1]
d = X_train.shape[1]

# sigma = 1113.379020273871
sigma = np.median(pdist(X_train[
    np.random.choice(X_train.shape[0], 1000), :]))
# sigma = 1. / np.sqrt(2 * 0.00728932)
print 'sigma: ', sigma
# print y.shape

lbda = 10. / n
batch = 20
block = 1
T = 10000
print 'T:', T, 'batch:', batch, 'block:', block, 'lbda:', lbda
cond = 1e-10

for eta0 in [8, 4, 2, 1, .5, .25, .125]:
    A = SoftMaxLoss()
    # A = HingeLoss()
    B = DecomposableGaussian(np.eye(p), sigma)
    # B = DecomposableGaussian(np.eye(p), np.sqrt(sigma))
    # B = DecomposableGaussian(np.cov(y_train.T), sigma)
    C = InverseScaling(eta0, .0001)

    estimator = DSOVK(A, B, C, lbda, T, batch, block, 0, cond)
    ts = time.time()
    estimator.fit(X_train, y_train)
    te = time.time()
    print 'Train time: ', te - ts

    # ts = time.time()
    pred_train = estimator.predict(X_train)
    # te = time.time()
    # print 'Pred time: ', te - ts

    # print lb.inverse_transform(pred_train)
    # print lb.inverse_transform(y_train)

    score = 1 - (lb.inverse_transform(pred_train) ==
                 lb.inverse_transform(y_train)).sum() * (1. / y_train.shape[0])
    print score * 100, "%"

    ts = time.time()
    pred_test = estimator.predict(X_test)
    te = time.time()
    print 'Pred time: ', te - ts

    print lb.inverse_transform(pred_test)
    print lb.inverse_transform(y_test)

    score = 1 - (lb.inverse_transform(pred_test) ==
                 lb.inverse_transform(y_test)).sum() * (1. / y_test.shape[0])
    print score * 100, "%"
