#!/usr/bin/env python

"""

Gauss-Kronrod 7-15 automatic adaptive quadratures

	* total of 15 function evaluations for each subinterval

	The integral is then estimated by the Kronrod rule K15
	and the error can be estimated as |G7−K15|


	Gauss nodes					Weights

	±0.949107912342759		∗	0.129484966168870
	±0.741531185599394		∗	0.279705391489277
	±0.405845151377397		∗	0.381830050505119
	 0.000000000000000		∗	0.417959183673469

	Kronrod nodes				Weights

	±0.991455371120813			0.022935322010529
	±0.949107912342759		∗	0.063092092629979
	±0.864864423359769			0.104790010322250
	±0.741531185599394		∗	0.140653259715525
	±0.586087235467691			0.169004726639267
	±0.405845151377397		∗	0.190350578064785
	±0.207784955007898			0.204432940075298
	 0.000000000000000		∗	0.209482141084728

"""

__author__  = "LJ Brown"
__file__ = "GaussKronrodQuadrature.py"

import numpy as np


def G7K15(a, b):
	""" 

	Usage: 
		[InG, InK, err] = GKK15(f, a, b) 

	InG            -- Gauss intergral approximation
	InK            -- Kronrod intergral approximation
	nf = 15 	   -- total of 15 function evaluations
	err = |G7−K15| -- approximation of error over interval [a,b]

	"""

	# K
	x0  = -0.991455371120813 
	x1  = 0.991455371120813 
	# GK
	x2  = -0.949107912342759 
	x3  = 0.949107912342759 
	# K
	x4  = -0.864864423359769 
	x5  = 0.864864423359769 
	# GK
	x6  = -0.741531185599394 
	x7  = 0.741531185599394 
	# K
	x8  = -0.586087235467691 
	x9  = 0.586087235467691 
	# GK
	x10 = -0.405845151377397 
	x11 = 0.405845151377397 
	# K
	x12 = -0.207784955007898
	x13 = 0.207784955007898
	# GK
	x14 = 0.000000000000000
	
	# K
	w0  = 0.022935322010529
	w1  = 0.022935322010529
	# GK
	w2  = 0.063092092629979
	w3  = 0.063092092629979
	# K
	w4  = 0.104790010322250
	w5  = 0.104790010322250
	# GK
	w6  = 0.140653259715525
	w7  = 0.140653259715525
	# K
	w8  = 0.169004726639267
	w9  = 0.169004726639267
	# GK
	w10 = 0.190350578064785
	w11 = 0.190350578064785
	# K
	w12 = 0.204432940075298
	w13 = 0.204432940075298
	# GK
	w14 = 0.209482141084728


	G_xs = [x2, x3,  x6, x7, x10, x11,  x14] 
	G_ws = [w2, w3,  w6, w7, w10, w11,  w14]

	K_xs = G_xs + [x0, x1, x4, x5, x8, x9, x12, x13] 
	K_ws = G_ws + [w0, w1, w4, w5, w8, w9, w12, w13]

	G_ws = [0.5*(b-a)*w for w in G_ws]
	G_xs = [0.5*(b-a)*x + 0.5*(a+b) for x in G_xs]
	K_ws = [0.5*(b-a)*w for w in K_ws]
	K_xs = [0.5*(b-a)*x + 0.5*(a+b) for x in K_xs]


	return G_ws, G_xs, K_ws, K_xs


class G7:

	def __init__(self):

		self.order = 7

	def get_weights_and_nodes(self, a, b):

		# project onto interval [a,b]
		gauss_weights, gauss_nodes, _, _ = G7K15(a, b)

		return gauss_weights, gauss_nodes

	def __str__(self):
		# "gauss-7"
		return "G7"

	def get_method_name(self):
		return "G7K15Quadrature"

class K15:

	def __init__(self):

		self.order = 15

	def get_weights_and_nodes(self, a, b):

		# project onto interval [a,b]
		_, _, kronrod_weights, kronrod_nodes = G7K15(a, b)

		return kronrod_weights, kronrod_nodes

	def __str__(self):
		# "gauss-kronrod-15"
		return "K15"

	def get_method_name(self):
		return "G7K15Quadrature"


class G7K15Quadrature:


	def __init__(self):
		pass

		
	def get_G7(self):
		return G7()

	def get_K15(self):
		return K15()

	def get_method_name(self):
		return "G7K15Quadrature"



