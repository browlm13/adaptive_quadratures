#!/usr/bin/env python

"""

Gauss-Lobatto 5 Quadrature

	* total of 5 function evaluations for each subinterval

	Gauss-Lobatto nodes			Weights

	±1.000000000000000		∗	0.100000000000000
	±0.654653670707977		∗	0.544444444444444
	 0.000000000000000		∗	0.711111111111111


"""

__author__  = "LJ Brown"
__file__ = "lobatto5.py"

import numpy as np

def lobatto5(f, a, b):
	""" Usage: [In,nf] = lobatto5(f, a, b) """

	# nodes for the legendre polynomial on the unit interval
	x0  = -1.000000000000000
	x1  =  1.000000000000000
	x2  = -0.654653670707977
	x3  =  0.654653670707977
	x4  =  0.000000000000000
	
	# weights for the legendre polynomial on the unit interval
	w0  =  0.100000000000000
	w1  =  0.100000000000000 
	w2  =  0.544444444444444
	w3  =  0.544444444444444
	w4  =  0.711111111111111

	# unit weights and nodes [-1,1]
	xs = np.array([x0, x1, x2, x3, x4])
	ws = np.array([w0, w1, w2, w3, w4])

	# project onto interval [a,b]
	xs = 0.5*(b-a)*xs + 0.5*(a+b)
	ws = 0.5*(b-a)*ws

	# compute integral approximation
	In = sum( ws * f(xs) )

	# record the number of function calls for f as nf
	nf = 5

	return [In, nf]
	