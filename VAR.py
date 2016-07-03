
# coding: utf-8

# In[1]:

# get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.kernel_approximation import RBFSampler
import pyov2sgd


# In[2]:

p = 7
order = 5
# np.random.seed(0)
A = np.zeros((p, p, order))
n_interraction = 10
eig_wrong = True
while (A != 0).sum() != n_interraction or eig_wrong:
#     A = np.eye(p)
    i1 = np.random.randint(0, p)
    i2 = np.random.randint(0, p)
    op = np.random.randint(0, order)
    A[i1, i2, op] = np.random.normal(0, 0.04)
    eig_wrong = False
    for i in range(order):
        eig_wrong = eig_wrong or (np.abs(np.linalg.eigvals(A[:, :, i])) >= 1).any()
        
# for i in range(p):
#     A[i, :] = A[i, :] / np.linalg.norm(A[i, :])
# A = A / np.linalg.norm(A)
# A = np.eye(5)


# In[3]:

plt.figure(figsize=(order * 2, 2))
for i in range(order):
    print(np.linalg.eigvals(A[:, :, i]))
    plt.subplot(1, order, i + 1)
    plt.pcolor(A[:, :, i], cmap='Greys')

plt.figure(figsize=(2, 2))
plt.pcolor(np.sum(A, axis=2), cmap='Greys')


# In[6]:

#### D = 5000
# gamma = .8
# np.random.seed(0)
# phi = RBFSampler(gamma=gamma, n_components=D, random_state=0)
# theta = 2 * np.random.normal(0, 0.04, (D, p) - 1

# A = np.eye(p)

T = 1000
serie = np.empty((T, p))
for i in range(order):
    serie[i, :] = np.random.normal(0, 1., p)
for t in range(1, T):
    for i in range(order):
        serie[t, :] = np.dot(serie[t - i - 1, :], A[:, :, i]) + np.random.normal(0, 1., p)
    


# In[7]:

plt.figure(figsize=(16, 8))
plt.plot(np.arange(0, T, 1), serie)


# In[8]:

X = serie[:-1, :].copy()
y = serie[1:, :].copy()


# In[ ]:

G = np.eye(p)
# G = np.sum(A, axis=2)
# G = np.dot(G, G.T)
eta0 = 1.
lbda = 0.
nu = 0.
batch = X.shape[0]
block = 25
cond = 1e-10
cap = 10
T = 1
L = pyov2sgd.RidgeLoss()
B = pyov2sgd.DecomposableGaussian(G, 1000.)
C = pyov2sgd.InverseScaling(eta0, 0, 0)

estimator = pyov2sgd.BDSOVK(L, B, C, lbda, nu, T, p, batch, block, cap, cond)
# estimator = pyov2sgd.DSOVK(L, B, C, lbda, nu, T, p, batch, block, cap, cond)
start = time.time()
estimator.fit_dense(X, y)
stop = time.time()
print (3, 'T fit: ', stop - start)


# In[39]:

yp = estimator.predict_dense(serie[:-1, :])
np.mean((yp - serie[:-1, :]) ** 2)


# In[40]:

plt.figure(figsize=(16, 4))
plt.subplot(1, 2, 1)
plt.plot(np.arange(0, 999, 1), serie[:-1, :])

plt.subplot(1, 2, 2)
plt.plot(np.arange(0, 999, 1), yp)


# In[42]:

plt.pcolor(estimator.B(), cmap='Greys')


# In[43]:

plt.pcolor(np.sum(A, axis=2), cmap='Greys')


# In[ ]:



