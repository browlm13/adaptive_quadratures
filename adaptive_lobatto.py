#!/usr/bin/env python

"""

	Adaptive Lobatto

	use:
		from adaptive_lobatto import adaptive_lobatto
		In, nf = adaptive_lobatto(f, a, b, tol)


"""

__author__  = "LJ Brown"
__file__ = "adaptive_lobatto.py"

import math
import numpy as np

def magnitude(x):
	"""order of magnitude of x"""
	if x == 0.0:
		return int(math.log10(np.finfo(np.float32).eps))
	return int(math.log10(x))

class NFHashTable:

	def __init__(self, f):
		""" hash table for checking if nodes have already been evaluated f(xi). """

		# create empty hash table
		self.hash_table = {}

		self.f = f

	def reuse_evaluation(self, x):

		if x in self.hash_table.values():
			# if node has already been evaluated
			# return value from hash table
			# additional function evaluations: 0
			return self.hash_table[x]

		# otherwise find value by calling function
		# additional function evaluations: 1
		value = self.f(x)

		# add value to hash table
		self.hash_table[x] = value
		
		return value

	def get_nf(self):
		""" Return the total number of function calls performed """
		return len(self.hash_table)

class IntervalTree:
	""" Interval Tree for managing subinterval division. Tries to select high error regions to divide further."""

	class Node: 
		""" Binary tree node """
		
		def __init__(self, data_dict): 
			self.data_dict = data_dict

			# node children
			self.left = None
			self.right = None


	def __init__(self, quad, A, B):
		self.A = A
		self.B = B
		self.quad = quad

		self.create_root()

	def create_root(self):

		# compute first approximation
		In1, In2, local_error = self.quad(self.A, self.B)

		data_dict = {
			'a' : self.A,
			'b' : self.B,
			'In1' : In1,
			'In2' : In2,
			'local_error' : local_error
		}

		self.root = self.Node(data_dict)

	def divide(self):

		# divide subinterval with highest local error
		max_error_node = self.max_leaf_by_key('local_error')
		self._divide(max_error_node)


	def global_error(self): 
		""" Gloabl error is the sum of the local errors in leaf nodes """

		g_error = self.sum_leaf_entries(self.root, 'local_error')

		return g_error

	def Iapprox(self): 
		""" Integral approximation is the sum of the higher order method approximations """

		return self.sum_leaf_entries(self.root, 'In2')

	def leafs(self):
		""" return list of tree leaf nodes """
		return self._get_leafs(self.root)

	def _divide(self, node):

		# create children by splitting the interval in two
		a, b = node.data_dict['a'], node.data_dict['b']
		c = a + (b-a)/2

		# compute integral approximation for each node using lower order and higher order methods for each
		a_left, b_left = a, c
		a_right, b_right = c, b

		In1_left, In2_left, local_error_left = self.quad(a_left, b_left)
		In1_right, In2_right, local_error_right = self.quad(a_right, b_right)

		# create nodes and add them to tree
		data_dict_l = {
			'a' : a_left,
			'b' : b_left,
			'In1' : In1_left,
			'In2' : In2_left,
			'local_error' : local_error_left

		}

		data_dict_r = {
			'a' : a_right,
			'b' : b_right,
			'In1' : In1_right,
			'In2' : In2_right,
			'local_error' : local_error_right
		}

		node.left = self.Node(data_dict_l)
		node.right = self.Node(data_dict_r)

	def max_leaf_by_key(self, entry_key):

		max_node = self.leafs()[0]
		self.max_entry = max_node.data_dict[entry_key]
		for ln in self.leafs():
			if ln.data_dict[entry_key] >= max_node.data_dict[entry_key]:
				max_node = ln

		return max_node


	def sum_leaf_entries(self, node, entry_key):

		if node is None: 
			return 0

		if ( node.left is None and node.right is None ): 
			return node.data_dict[entry_key]

		else: 
			return self.sum_leaf_entries(node.left, entry_key) + self.sum_leaf_entries(node.right, entry_key)

	def count_leaf_nodes(node): 

		if node is None: 
			return 0 

		if(node.left is None and node.right is None): 
			return 1 

		else: 
			return count_leaf_nodes(node.left) + count_leaf_nodes(node.right) 

	def _get_leafs(self, node):
		""" private member - return list of leaf nodes """
		if node is not None: 

			if ( node.left is None and node.right is None ): 
				return [node]

			else: 
				return self._get_leafs(node.left) + self._get_leafs(node.right)

	def __len__(self):
		""" returns number of leaf nodes """
		return self.count_leaf_nodes()


class AdaptiveLobatto1213:

	def __init__(self):
		self.min_h = 0.01

	def lobatto12_weights_and_nodes(self,a,b):

		x0  = -1
		x1  = -0.9448992722228822234076
		x2  = -0.8192793216440066783486
		x3  = -0.6328761530318606776624
		x4  = -0.3995309409653489322644
		x5  = -0.1365529328549275548641
		x6  =  0.1365529328549275548641
		x7  =  0.3995309409653489322644
		x8  =  0.6328761530318606776624
		x9  =  0.8192793216440066783486
		x10 =  0.9448992722228822234076
		x11 =  1

		w0 = 0.01515151515151515151515
		w1 = 0.091684517413196130668
		w2 = 0.1579747055643701151647
		w3 = 0.212508417761021145358
		w4 = 0.2512756031992012802932
		w5 = 0.2714052409106961770003
		w6 = 0.2714052409106961770003
		w7 = 0.251275603199201280293
		w8 = 0.212508417761021145358
		w9 = 0.1579747055643701151647
		w10 = 0.0916845174131961306683
		w11 = 0.01515151515151515151515

		# unit weights and nodes [-1,1]
		xs = np.array([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11])
		ws = np.array([w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11])

		# project onto interval [a,b]
		xs = 0.5*(b-a)*xs + 0.5*(a+b)
		ws = 0.5*(b-a)*ws

		return ws, xs

	def lobatto13_weights_and_nodes(self,a,b):

		x0 = -1
		x1 = -0.9533098466421639118969
		x2 = -0.8463475646518723168659
		x3 = -0.6861884690817574260728
		x4 = -0.4829098210913362017469
		x5 = -0.2492869301062399925687
		x6 = 0
		x7 = 0.2492869301062399925687
		x8 = 0.4829098210913362017469
		x9 = 0.6861884690817574260728
		x10 = 0.8463475646518723168659
		x11 = 0.9533098466421639118969
		x12 = 1

		w0 = 0.01282051282051282051282
		w1 = 0.0778016867468189277936
		w2 = 0.1349819266896083491199
		w3 = 0.1836468652035500920075
		w4 = 0.2207677935661100860855
		w5 = 0.2440157903066763564586
		w6 = 0.251930849333446736044
		w7 = 0.2440157903066763564586
		w8 = 0.220767793566110086086
		w9 = 0.1836468652035500920075
		w10 = 0.1349819266896083491199
		w11 = 0.077801686746818927794
		w12 = 0.01282051282051282051282

		# unit weights and nodes [-1,1]
		xs = np.array([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12])
		ws = np.array([w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12])

		# project onto interval [a,b]
		xs = 0.5*(b-a)*xs + 0.5*(a+b)
		ws = 0.5*(b-a)*ws

		return ws, xs

	def adaptive_quad(self, f, a, b, tol=1e-10, other_args={}):
		""" Wrapper method """
		try: 
			del other_args['Itrue']
		except:
			pass
		return self.adaptive_quadrature(f, a, b, tol=tol, **other_args)

	def adaptive_quadrature(self, f, a, b, tol=1e-10, maxit=1000, output=False):

		# init members
		self.f = f
		self.tol = tol
		self.output = output

		# check inputs
		if (b < a):
			raise ValueError('adaptive_quadrature error: b < a!')

		# create empty hash table
		self.hash_table = NFHashTable(self.f)

		# create Interval Tree
		self.interval_tree = IntervalTree(self.quad, a, b)

		for i in range(maxit):

			g_err = self.interval_tree.global_error()

			if magnitude(g_err) < magnitude(self.tol):

				if self.output:
					print("tolerence threshold met.")
					print("number of subintervals used: %s" %  len(self.interval_tree))

				return [self.interval_tree.Iapprox(), self.hash_table.get_nf()]

			# otherwise subdivide and try again
			self.interval_tree.divide()
			

		if self.output:
			print("failure to meet tolernce")
			print("number of subintervals used: %s" %  len(self.interval_tree))

		return [self.interval_tree.Iapprox(), self.hash_table.get_nf()]

	def quad(self, a, b):
		""" 

		Usage: 
			[In1, In2, err] = self.quad(f, a, b) 

		In1            -- Intergral approximation from low order method
		In2            -- Intergral approximation from high order method
		err 		   -- approximation of error over interval [a,b]

		"""

		ws1, xs1 = self.lobatto12_weights_and_nodes(a,b) 
		ws2, xs2 = self.lobatto13_weights_and_nodes(a,b) 

		# use hash table to retreive any previously computed values
		# to  minimize number of function calls -- nf -- hash_table.get_nf()
		fxs1 = []
		fxs2 = []
		for x in xs1:
			fx = self.hash_table.reuse_evaluation(x)
			fxs1.append(fx)

		for x in xs2:
			fx = self.hash_table.reuse_evaluation(x)
			fxs2.append(fx)

		# calculate integral approximations, reusing all function evaluations
		In1 = sum([w*fx for w,fx in zip(ws1,fxs1)]) 
		In2 = sum([w*fx for w,fx in zip(ws2,fxs2)])

		#
		# error approximation
		# 

		err = self.error_estimate(abs(b-a), In1, In2)

		return [In1, In2, err]

	def error_estimate(self, h, In1, In2):
		""" chosen error estimate using heuristics """

		d = abs(In2 - In1)
		q = 3.38

		if h < 1:
			return d * h**q

		return d / h * 2


	def short_string(self):
		return "adaptive_lobatto"

	def __str__(self):			
		return self.short_string()

def adaptive_lobatto(f, a, b, tol):
	""" function [In,nf] = adaptive_lobatto(f, a, b, tol) """
	al = AdaptiveLobatto1213()
	return al.adaptive_quad(f, a, b, tol=tol)

