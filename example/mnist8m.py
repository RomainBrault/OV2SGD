import numpy as np
import pyov2sgd
import time

from scipy.spatial.distance import pdist
from sklearn.utils import shuffle

base = 'datasets/mnist/'

X = pyov2sgd.read_sparse_char2double(base + "mnist_X.bin")
y = pyov2sgd.read_sparse_char2double(base + "mnist_y.bin")

X_test = pyov2sgd.read_sparse_char2double(base + "mnist_X_test.bin")
y_test = pyov2sgd.read_sparse_char2double(base + "mnist_y_test.bin")

# X /= 255
# X_test /= 255

# sigma = 1113.379020273871
sigma = np.median(pdist(shuffle(X.toDense())[:1000, :]))
# sigma = 1. / np.sqrt(2 * 0.00728932)
print ('sigma: ', sigma)
# print y.shape

# n = X.shape[1]
# p = y.shape[0]
n = 60000
p = 10

lbda = 10. / n
batch = 2000
block = 1000
T = 30
cond = 1e-10

A = pyov2sgd.SoftMaxLoss()
# A = pyov2sgd.HingeLoss()
B = pyov2sgd.DecomposableGaussian(np.eye(p), sigma)
# B = pyov2sgd.DecomposableGaussian(np.cov(y.toDense().T), sigma)
eta0 = 2.
eta1 = .00001
C = pyov2sgd.InverseScaling(eta0, eta1)

print ('T:', T, 'batch:', batch, 'block:', block, 'lbda:', lbda, 'C',
       (eta0, eta1))

t = T
# for t in range(120, T, 10):
estimator = pyov2sgd.DSOVK(A, B, C, lbda, t, p, batch, block, 1, cond)

# print X
start = time.time()
estimator.fit_sparse(X, y)
stop = time.time()
print ('T fit: ', stop - start)

start = time.time()
pred = estimator.predict_sparse(X_test)
stop = time.time()
print ('T pred: ', stop - start)
# print type(pred)
# print pred.shape
# # print pred
# X /= 10
# print X
# pred = np.zeros((5, 5))
print (np.sum(pred.argmax(axis=1) == y_test.toDense().argmax(axis=1)) /
       float(pred.shape[0]) * 100)
