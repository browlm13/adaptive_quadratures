#!/usr/bin/env python

"""

	Gauss-Lobatto automatic adaptive quadratures

"""

__author__  = "LJ Brown"
__file__ = "LobattoQuadrature.py"

# internal
import warnings

# external
import numpy as np

# dir
from newton_interp import *
from newton import *


def L(n):
	"""
		Generate the Legendre polynomial function of order n recusrivley

		P(0,X) = 1
		P(1,X) = X
		P(N,X) = ( (2*N-1)*X*P(N-1,X)-(N-1)*P(N-2,X) ) / N
	"""
	if (n==0):
		return lambda x: 1.0

	elif (n==1):
		return lambda x: x

	else:
		return lambda x: ( (2.0*n-1.0) * x * L(n-1)(x)-(n-1) * L(n-2)(x) ) / n

def dL(n):
	"""
		Generate derivative function of the Legendre polynomials of 
		order n recursivley
		
		P'(0,X) = 0
		P'(1,X) = 1
		P'(N,X) = ( (2*N-1)*(P(N-1,X)+X*P'(N-1,X)-(N-1)*P'(N-2,X) ) / N
	"""

	#[TODO]: allow evaluation at 0
	if (n==0):
		return lambda x: 0.0

	elif (n==1):
		return lambda x: 1.0

	else:
		return lambda x: (n/(x**2-1.0))*(x*L(n)(x)-L(n-1)(x))

warnings.filterwarnings("ignore")
def coef_approximation(p, order):
	"""
		Get coefficients approximation of polynomial
	"""

	# maintain parity of order +1
	n = 50 + order +1
	r =  1
	xs = np.linspace(-r, r, num=n)
	ys = p(xs)

	# [TODO]: fix coeffients method
	# replace with 'c = coeffients(xs, ys)'
	degree = n 
	c = np.polyfit(xs,ys,degree)

	return c

def polynomial_derivative(coefficients):

	# compute coefficients for first derivative of p with coefficients c
	# [TODO]: fix own method from Hw 3
	c_prime = np.polyder(coefficients)

	return c_prime

def ddL(n):
	"""

		Generate second derivative of the Lengendre polynomials of
		order n recursivley

			P"(0,X) = 0
			P"(1,X) = 0
			P"(N,X) = ( (2*N-1)*(2*P'(N-1,X)+X*P"(N-1,X)-(N-1)*P'(N-2,X) ) / N
	"""

	if (n==0):
		return lambda x: 0.0

	elif (n==1):
		return lambda x: 0.0

	else:

		# approximate by fitting polynomial and taking derivatives
		c_om1 = coef_approximation(L(n), n)
		c_prime = polynomial_derivative(c_om1)
		c_double_prime = polynomial_derivative(c_prime)

		# [TODO]: fix own method from Hw 3
		return lambda x: np.polyval(c_double_prime, x)

def unit_lobatto_nodes(order, tol=1e-15, output=True):
	"""
		Find an approximation for roots of the derivative of the legendre polynomial 
		of given order on the unit interval [-1, 1].
	"""

	roots=[]


	# The polynomials are alternately even and odd functions
	# so evaluate only half the number of roots.
	# lobatto polynomial is derivative of legendre polynomial of order n-1
	order = order-1
	for i in range(1,int(order/2) +1):

		# initial guess, x0, for ith root 
		# the approximate values of the abscissas.
		# these are good initial guesses
		#x0=np.cos(np.pi*(i-0.25)/(order+0.5)) 
		x0=np.cos(np.pi*(i+0.1)/(order+0.5))  # not sure why this inital guess is better

		# call newton to find the roots of the lobatto polynomial
		Ffun, Jfun = dL(order), ddL(order) 
		ri, _ = newton( Ffun, Jfun, x0 )

		roots.append(ri)

	# remove roots close to zero
	cleaned_roots = []
	tol = 1e-08
	for r in roots:
		if abs(r) >= tol:
			cleaned_roots += [r]
	roots = cleaned_roots

	# use symetric properties to find remmaining roots
	# the nodal abscissas are computed by finding the 
	# nonnegative zeros of the Legendre polynomial pm(x) 
	# with Newton’s method (the negative zeros are obtained from symmetry).
	roots = np.array(roots)
	
	# add -1 and 1 to tail ends
	# check parity of order + 1
	# even. no center 
	if (order + 1) % 2==0:
		roots = np.concatenate( ([-1.0], -1.0*roots, roots[::-1], [1.0]) )

	# odd. center root is 0.0
	else:
		roots = np.concatenate( ([-1.0], -1.0*roots, [0.0], roots[::-1], [1.0] ) )

	return roots

def unit_lobatto_weights_and_nodes(order):
	"""
		Find weights for the lobatto polynomial
		of given order on the unit interval [-1, 1].

		Ai = 2 / [ (1 - xi^2)* (p'n+1(xi))^2 ]  -- Gauss Legendre Weights
		wi = 2/(n(n-1) * Pn-1(xi)^2)
	"""

	# find roots of legendre polynomial  on unit interval
	nodes = unit_lobatto_nodes(order)

	# calculate weights for unit interval
	# wi = 2/(n(n-1) * Pn-1(xi)^2)
	weights = 2.0/( (order*(order-1)) * L(order-1)(nodes)**2 )

	return weights, nodes

def project_weights_and_nodes(a, b, unit_weights, unit_nodes):
	"""
		Given unit weights and nodes on interval [-1,1] map to interval [a,b].
	"""

	# project onto interval [a,b]
	nodes = 0.5*(b-a)*unit_nodes + 0.5*(a+b)
	weights = 0.5*(b-a)*unit_weights

	return weights, nodes

class LobattoQuadrature:

	def __init__(self, order):

		self.order = order
		self.unit_weights, self.unit_nodes = unit_lobatto_weights_and_nodes(self.order)

	def get_weights_and_nodes(self, a, b):

		# project onto interval [a,b]
		weights, nodes = project_weights_and_nodes(a, b, self.unit_weights, self.unit_nodes)

		return weights, nodes

	def __str__(self):
		# "gauss-lobatto-"
		return "L" + str(self.order)

	def get_method_name(self):
		return "LobattoQuadrature"
