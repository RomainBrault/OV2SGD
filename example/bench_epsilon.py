import io
import urllib
import cvxpy as cp
import scipy.sparse as sp
import numpy as np
import numpy.linalg as LA
import epopt as ep


data = "http://epopt.s3.amazonaws.com/mnist.npz"
mnist = np.load(io.BytesIO(urllib.urlopen(data).read()))


# Multiclass classification
def one_hot(y, k):
    m = len(y)
    return sp.coo_matrix((np.ones(m),
                         (np.arange(m), y)), shape=(m, k)).todense()


# Get solution and compute train/test error
def error(x, y):
    return 1 - np.sum(x == y) / float(len(x))


def multiclass_hinge_loss(Theta, X, y):
    k = Theta.size[1]
    Y = one_hot(y, k)
    return (cp.sum_entries(cp.max_entries(X * Theta + 1 - Y, axis=1)) -
            cp.sum_entries(cp.mul_elemwise(X.T.dot(Y), Theta)))


def median_dist(X):
    """Compute the approximate median distance by sampling pairs."""
    k = 1 << 20  # 1M random points
    i = np.random.randint(0, X.shape[0], k)
    j = np.random.randint(0, X.shape[0], k)
    return np.sqrt(np.median(np.sum((X[i, :] - X[j, :])**2, axis=1)))


def pca(X, dim):
    """Perform centered PCA."""
    X = X - X.mean(axis=0)
    return LA.eigh(X.T.dot(X))[1][:, -dim:]

# PCA and median trick
np.random.seed(0)
V = pca(mnist["X"], 50)
X = mnist["X"].dot(V)
sigma = median_dist(X)


y = mnist["Y"].ravel()
ytest = mnist["Ytest"].ravel()

# Random features
n = 4000
W = np.random.randn(X.shape[1], n) / sigma
b = np.random.uniform(0, 2 * np.pi, n)
X = np.cos(X.dot(W) + b)
Xtest = np.cos(mnist["Xtest"].dot(V).dot(W) + b)

# Parameters
m, n = X.shape
k = 10
Theta = cp.Variable(n, k)
lam = 10

# Form problem with CVXPY and solve with Epsilon
f = ep.multiclass_hinge_loss(Theta, X, y) + lam * cp.sum_squares(Theta)
prob = cp.Problem(cp.Minimize(f))
ep.solve(prob, verbose=True)

# Get solution and compute train/test error
Theta0 = np.array(Theta.value)
print "Train error:", error(np.argmax(X.dot(Theta0), axis=1), y)
print "Test error:", error(np.argmax(Xtest.dot(Theta0), axis=1), ytest)
