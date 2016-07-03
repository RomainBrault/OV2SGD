import numpy as np
import pyov2sgd
import time
import operalib
import matplotlib.pyplot as plt

from sklearn.cross_validation import KFold

# ('T:', 60, 'batch:', 2000, 'block:', 1000, 'lbda:', 0.00016666666666666666, 'C', (2.0, 0.0001))
# -> 98.03% in 1139.3162949085236s

base = 'datasets/mnist/'

X = pyov2sgd.read_sparse_char2double(base + "mnist_X.bin").toDense()
y = pyov2sgd.read_sparse_char2double(base + "mnist_y.bin").toDense()

X_test = pyov2sgd.read_sparse_char2double(base + "mnist_X_test.bin").toDense()
y_test = pyov2sgd.read_sparse_char2double(base + "mnist_y_test.bin").toDense()

Xt = np.vstack((X, X_test))
yt = np.vstack((y, y_test))


# X_train, X_test, y_train, y_test = \
#     cross_validation.train_test_split(Xt, yt, test_size=0.2, random_state=0)

# X /= 255
# X_test /= 255

# n = X.shape[1]
# p = y.shape[0]
n = 60000
p = 10

lbda = 10. / n
batch = 2000
block = 1000
T = 30
cond = 1e-10

np.random.seed(0)
kf = KFold(Xt.shape[0], n_folds=7, random_state=0)

m1 = np.empty((11, 7))
m2 = np.empty((11, 7))
m3 = np.empty((11, 7))

d1 = np.empty((11, 7))
d2 = np.empty((11, 7))
d3 = np.empty((11, 7))

i = 0
for T in np.hstack((1, np.logspace(np.log10(3), np.log10(60), 10).astype(np.int))):
    print ('T=', T)
    j = 0
    for train, test in kf:
        # ORFF model
        A = pyov2sgd.RidgeLoss()
        G = np.eye(p - 1)
        B = pyov2sgd.DecomposableGaussian(G, 1. / np.sqrt(2 * 0.031))
        eta0 = .25
        eta1 = .01
        C = pyov2sgd.InverseScaling(eta0, eta1)

        estimator = pyov2sgd.DSOVK(A, B, C, lbda, T, p - 1,
                                   batch, block, T, cond)
        start = time.time()
        estimator.fit_dense(Xt[train, :] / 255, pyov2sgd.sencode(yt[train, :]))
        stop = time.time()
        print (3, 'T fit: ', stop - start)
        d1[i, j] = stop - start

        start = time.time()
        pred = estimator.predict_dense(Xt[test, :] / 255)
        stop = time.time()
        score = (np.sum(pyov2sgd.sdecode(pred) == yt[test, :].argmax(axis=1)) /
                 float(pred.shape[0]) * 100)
        print (3, 'T pred: ', stop - start, score)
        m1[i, j] = score
        j = j + 1
    i = i + 1

i = 0
for N in np.logspace(0, np.log10(15000), 10):
    print ('N=', N)
    j = 0
    for train, test in kf:
        # OVK model
        subsample = int(N)
        G = np.eye(p - 1)
        start = time.time()
        regr_1 = operalib.Ridge('DGauss', gamma=0.031, lbda=10. / n, A=G)
        regr_1.fit(Xt[train, :][:subsample, :] / 255,
                   pyov2sgd.sencode(yt[train, :][:subsample, :]))
        stop = time.time()
        print(2, "T fit: ", time.time() - start)
        d2[i, j] = stop - start

        start = time.time()
        pred = regr_1.predict(Xt[test, :] / 255)
        score = (np.sum(pyov2sgd.sdecode(pred) == yt[test, :].argmax(axis=1)) /
                 float(pred.shape[0]) * 100)
        print(2, "T pred: ", time.time() - start, score)
        m2[i, j] = score
        j = j + 1
    i = i + 1

i = 0
for T in np.hstack((1, np.logspace(np.log10(3), np.log10(60), 10).astype(np.int))):
    print ('T=', T)
    j = 0
    for train, test in kf:
        # skewed chi2 ORFF model
        A = pyov2sgd.SoftMaxLoss()
        G = np.eye(p)
        B = pyov2sgd.DecomposableSkewedChi2(G, 700.)
        eta0 = 2.
        eta1 = .00001
        C = pyov2sgd.InverseScaling(eta0, eta1)

        estimator = pyov2sgd.DSOVK(A, B, C, lbda, T, p, batch, block, T, cond)

        start = time.time()
        estimator.fit_dense(Xt[train, :], yt[train, :])
        stop = time.time()
        print (1, 'T fit: ', stop - start)
        d3[i, j] = stop - start

        start = time.time()
        pred = estimator.predict_dense(Xt[test, :])
        stop = time.time()
        score = (np.sum(pred.argmax(axis=1) == yt[test, :].argmax(axis=1)) /
                 float(pred.shape[0]) * 100)
        print (1, 'T pred: ', stop - start, score)
        m3[i, j] = score
        j = j + 1
    i = i + 1

t1 = np.hstack((1, np.logspace(np.log10(3), np.log10(60), 10).astype(np.int))) * 2000
t2 = np.hstack((1, np.logspace(np.log10(3), np.log10(60), 10).astype(np.int))) * 2000
t3 = np.hstack((1, np.logspace(np.log10(3), np.log10(60), 10).astype(np.int))) * 2000

print m1
print m2
print m3

print d1
print d2
print d3

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('MNIST')
orff, = ax.plot(t1, m1.mean(axis=1), lw=2)
ovk, = ax.plot(t2, m2.mean(axis=1), lw=2)
orffsk, = ax.plot(t3, m3.mean(axis=1), lw=2)
epoch = ax.axvline(60000, color='k', linestyle='--')
# ax.set_xscale('log')
ax.set_xlim([1900, 130000])
ax.set_ylim([90, 100])
plt.xlabel('Data seen (N)')
plt.ylabel('Test accuracy (%)')

axt = ax.twinx()
axt.set_ylabel('time (s)')
orfft, = axt.plot(t1, d1.mean(axis=1), lw=2, linestyle=':')
orfft, = axt.plot(t2, d2.mean(axis=1), lw=2, linestyle=':')
orfft, = axt.plot(t3, d3.mean(axis=1), lw=2, linestyle=':')
# axt.set_xscale('log')
axt.set_yscale('log', basey=2)
axt.set_xlim([1900, 130000])
axt.set_ylim([1, 1700])

# ax.grid()
axt.grid()

plt.legend([ovk, orff, orffsk, epoch], ['OVK-Gs', 'ORFF-Gs', 'ORFF-Sc2', '1 epoch'], loc=4)
plt.savefig('res_mnist.pdf', format='pdf', dpi=300)
plt.savefig('res_mnist.png', format='png', dpi=300)
plt.savefig('res_mnist.svg', format='svg', dpi=300)
plt.savefig('res_mnist.ps', format='ps', dpi=300)
