"""Doubly Stochastic gradient descent for operator-valued kernels."""

from _pyov2sgd import DecomposableGaussian, \
    RidgeLoss, HingeLoss, SoftMaxLoss, \
    InverseScaling, DSOVK

__all__ = ["DSOVK",
           "RidgeLoss", "HingeLoss", "SoftMaxLoss",
           "DecomposableGaussian",
           "InverseScaling"]
