#!/usr/bin/env python

"""

	Gauss-Legendre automatic adaptive quadratures

"""

__author__  = "LJ Brown"
__file__ = "GaussKronrodQuadrature.py"

# external
import numpy as np

# dir
from newton import *

def L(n):
	"""
		Generate the Legendre polynomial function of order n recusrivley
	"""

	if (n==0):
		return lambda x: x*0+1.0

	elif (n==1):
		return lambda x: x

	else:
		return lambda x: ( (2.0*n-1.0) * x * L(n-1)(x)-(n-1) * L(n-2)(x) ) / n

def dL(n):
	"""
		Generate derivative function of the Legendre polynomials of 
		order n recursivley.
	"""

	if (n==0):
		return lambda x: x*0

	elif (n==1):
		return lambda x: x*0+1.0

	else:
		# (1 − x2)pn′(x) = n[−xpn(x) + pn−1(x)]
		return lambda x: (n/(x**2-1.0))*(x*L(n)(x)-L(n-1)(x))

def unit_lengendre_roots(order, tol=1e-15, output=False):
	"""
		Find an approximation for roots of the legendre polynomial
		of given order on the unit interval [-1, 1].
	"""

	roots=[]

	if output:
		print("Finding roots of Legendre polynomial of order ", order)

	# Polynomials alternate parity
	# so evaluate only half of number of roots
	for i in range(1,int(order/2) +1):

		# Initial guess, x0, for ith root 
		# the approximate values of the abscissas.
		# these are good initial guesses.
		x0=np.cos(np.pi*(i-0.25)/(order+0.5)) 

		# Call newton to find the roots of the legendre polynomial
		Ffun, Jfun = L(order), dL(order)
		ri, _ = newton( Ffun, Jfun, x0 )

		roots.append(ri)

	# Use symetric properties to find remmaining roots
	# the nodal abscissas are computed by finding the 
	# nonnegative zeros of the Legendre polynomial pm(x) 
	# with Newton’s method (the negative zeros are obtained from symmetry).
	roots = np.array(roots)

	# even. no center
	if order % 2==0:
		roots = np.concatenate( (-1.0*roots, roots[::-1]) )

	# odd. center root is 0.0
	else:
		roots = np.concatenate( (-1.0*roots, [0.0], roots[::-1]) ) 

	return roots

def unit_gauss_weights_and_nodes(order):
	"""
	Find weights for the roots of the legendre polynomial 
	of given order on the unit interval [-1, 1].

	Ai = 2 / [ (1 - xi^2)* (p'n+1(xi))^2 ]  -- Gauss Legendre Weights
	"""

	# find roots of legendre polynomial  on unit interval
	nodes = unit_lengendre_roots(order)

	# calculate weights for unit interval
	weights = 2.0/( (1.0-nodes**2) * dL(order)(nodes)**2 )

	return weights, nodes

def project_weights_and_nodes(a, b, unit_weights, unit_nodes):
	"""
 		Given unit weights and nodes on interval [-1,1] map to interval [a,b]
	"""

	# project onto interval [a,b]
	nodes = 0.5*(b-a)*unit_nodes + 0.5*(a+b)
	weights = 0.5*(b-a)*unit_weights

	return weights, nodes

def gauss_quadrature(f, a, b, order, weights=None, nodes=None):

	# if weights and nodes are None, compute them for the interval [a,b]
	if weights is None: 

		assert nodes is None

		# find weights for the legendre polynomial on the unit interval
		unit_weights, unit_nodes = unit_gauss_weights_and_nodes(order)

		# project onto interval [a,b]
		weights, nodes = project_weights_and_nodes(a, b, unit_weights, unit_nodes)

	# compute integral approximation
	Iapprox = sum( weights * f(nodes) )

	# record the number of function calls for f as nf
	nf = order

	return Iapprox, nf


def composite_gauss_quadrature(f, a, b, order, m):

	# m in number of sub intervals

	# check inputs
	if (b < a):
		raise ValueError('composite_gauss_quadrature error: b < a!')
	if (m < 1):
		raise ValueError('composite_gauss_quadrature error: m < 1!')

	# set up subinterval width
	h = 1.0*(b-a)/m

	# initialize results
	Imn = 0.0
	nf = 0

	# compute the unit weights and nodes for the lobatto quadrature 
	# of given order
	unit_weights, unit_nodes = unit_gauss_weights_and_nodes(order)

	# [TODO]: reuse calculation of end nodes and weights
	# iterate over subintervals
	for i in range(m):

		# define subintervals start and stop points
		ai, bi = a+i*h, a+(i+1)*h

		# project onto interval [a,b]
		weights, nodes = project_weights_and_nodes(ai, bi, unit_weights, unit_nodes)

		# call lobatto quadrature formula on this subinterval
		In, nlocal = gauss_quadrature(f, ai, bi*h, order, weights=weights, nodes=nodes)

		# increment outputs
		Imn += In
		nf  += nlocal

	return Imn, nf


class GaussLegendreQuadrature:

	def __init__(self, order):

		self.order = order

		self.unit_weights, self.unit_nodes = unit_gauss_weights_and_nodes(self.order)


	def get_weights_and_nodes(self, a, b):

		# project onto interval [a,b]
		weights, nodes = project_weights_and_nodes(a, b, self.unit_weights, self.unit_nodes)

		return weights, nodes

	def __str__(self):
		# "gauss-legendre-"
		return "G" + str(self.order)

	def get_method_name(self):
		return "GaussLegendreQuadrature"

