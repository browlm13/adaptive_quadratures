#!/usr/bin/env python3

"""
    File name: newton_interp.py
    Python Version: 3.6

        Construct a newton interpolating polynomial for f(x) using linspaced data points for nodes.
        Where f(x*) = 0 is the root finding form of the fixed point problem function g(x*) = x(*).

                                    f(x*) = g(x*) - x* = 0.

                                        (Plot Results)

    TODO: use chebyshev nodes

    L.J. Brown
    Math5315 @ SMU
    Fall 2018
"""

__filename__ = "newton_interp.py"
__author__ = "L.J. Brown"

# external libraries
import numpy as np

def coeffients(x, y):
    """ 
        Computes and returns the coeffients of the interpolating polynomial of degree len(x).
        refrence sources: ['https://stackoverflow.com/questions/14823891/newton-s-interpolating-polynomial-python']

        :param x: 1d numpy array of x datapoints.
        :param y: 1d numpy array of f(x) datapoints.
        :returns: 1d numpy array of coeffiencts for newton interpolating polynomial.
    """

    # ensure floating point datatypes
    x.astype(float)
    y.astype(float)

    # degree of interpolating polynomial
    n = len(x)

    # intitilize list of coeffients for interpolating polynomial to y values
    c = y.tolist()

    # compute coeffients
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            c[i] = float(c[i]-c[i-1])/float(x[i]-x[i-j])

    # return an array of polynomial coefficient, note: reverse order for np.polyval function
    return np.array(c[::-1])
