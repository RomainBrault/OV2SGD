"""Continuous one class svm."""

import numpy as np

from matplotlib import pyplot as plt


obs = np.concatenate((np.random.randn(100, 2),
                      1 + np.random.randn(300, 2)))


