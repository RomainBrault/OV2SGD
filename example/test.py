from pyov2sgd import *
from operalib import *
from scipy.spatial.distance import pdist, squareform
import numpy as np
import time
# A = np.random.rand(5, 5)
# print A
# ov2sgd.pyarray2mat(A)

# A = ov2sgd.mat2pyarray()
# print A
# A = ov2sgd.dec_gauss_rff(np.eye(5))
# A = ov2sgd.DecomposableGaussian(np.eye(5), 1., 0)
# print A[1]
# print A[2]

n = 1000 * 10
d = 28 * 28
p = 10
r = p

np.random.seed(0)
X = np.random.rand(n, d)
y = np.empty((n, p))
for i in xrange(p):
    y[:, i] = np.cos(0.5 * np.pi * np.linalg.norm(X, axis=1)) * \
        np.exp(-0.1 * np.pi * np.linalg.norm(X, axis=1))

# print y
y = y + 0.1 * np.random.randn(n, p)

ts = time.time()
sigma = .1 * d * np.median(pdist(X[:1000, :]))
te = time.time()
print 'sigma time: ', te - ts
print 'sigma: ', sigma
# print y.shape

lbda = 1e-6
batch = 20000
T = n / batch
# block = batch / 2
block = 1000
cond = 1e-6

A = RidgeLoss()
# B = DecomposableGaussian(np.cov(y.T), sigma)
B = DecomposableGaussian(np.eye(p), sigma)
C = InverseScaling(100., 1e-1)

# for i in xrange(T):
estimator = DSOVK(A, B, C, lbda, 1, batch, block, cond)
ts = time.time()
estimator.fit(X, y)
te = time.time()
print 'Train time: ', te - ts

ts = time.time()
pred = estimator.predict(X)
te = time.time()
print 'Pred time: ', te - ts
print (1 - np.linalg.norm(pred - y, axis=0) ** 2 /
       np.linalg.norm(y, axis=0) ** 2).mean()


Xt = np.random.rand(n, d)
yt = np.empty((n, p))
for i in xrange(p):
    yt[:, i] = np.cos(0.5 * np.pi * np.linalg.norm(Xt, axis=1)) * \
        np.exp(-0.1 * np.pi * np.linalg.norm(Xt, axis=1))

ts = time.time()
pred = estimator.predict(Xt)
te = time.time()
print 'Pred time: ', te - ts
print (1 - np.linalg.norm(pred - yt, axis=0) ** 2 /
       np.linalg.norm(yt, axis=0) ** 2).mean()

# # estimator = Ridge()
