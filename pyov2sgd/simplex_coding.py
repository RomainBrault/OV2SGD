"""Simplex coding module."""

import numpy


def scode(n):
    """Simplex coding operator."""
    return _scode_i(n - 1)


def _scode_i(n):
    """Simplex coding operator (internal).

    https://papers.nips.cc/paper/4764-multiclass-learning-with-simplex-coding.pdf
    """
    if n > 1:
        C1 = numpy.vstack((numpy.ones((1, 1)),
                           numpy.zeros((n - 1, 1))))
        C2 = numpy.vstack((numpy.full((1, n), -1. / n),
                           _scode_i(n - 1) * numpy.sqrt(1. - 1. / (n * n))))
        return numpy.hstack((C1, C2))
    if n == 1:
        return numpy.array([1, -1])
    if n < 1:
        raise "Dimension n should be at least one"""


def sencode(A):
    """Simplex coding encoder."""
    encoder = scode(A.shape[1]).T
    return numpy.dot(A, encoder)


def sdecode(A):
    """Simplex coding decoder."""
    decoder = scode(A.shape[1] + 1)
    return numpy.dot(A, decoder).argmax(axis=1)
