"""Doubly Stochastic gradient descent for operator-valued kernels."""
import sys

sys.path.append('/Users/Romain/OVRFF/pyov2sgd')

from simplex_coding import scode, sencode, sdecode

from _pyov2sgd import DecomposableGaussian, DecomposableGaussianB, \
    DecomposableSkewedChi2, DecomposableSkewedChi2B, \
    TransformableGaussian, DivergenceFreeGaussian, \
    MultitaskRidgeLoss, RidgeLoss, \
    SVRLoss, MultitaskSVRLoss, \
    HingeLoss, SoftMaxLoss, \
    InverseScaling, AverageScaling, \
    DSOVK, \
    read_sparse_char2double, \
    EigenSparseMatrix  # TSOVK, DivSOVK,


__all__ = ["DSOVK",  # "TSOVK", "DivSOVK",
           "RidgeLoss", "HingeLoss", "SoftMaxLoss", "SVRLoss",
           "MultitaskRidgeLoss", "MultitaskSVRLoss",
           "DecomposableGaussian", "DecomposableGaussianB",
           "DecomposableSkewedChi2", "DecomposableSkewedChi2B",
           "DivergenceFreeGaussian",
           "TransformableGaussian",
           "InverseScaling", "AverageScaling",
           "read_sparse_char2double",
           "EigenSparseMatrix",
           "scode", "sencode", "sdecode"]
