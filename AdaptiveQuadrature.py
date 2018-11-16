#!/usr/bin/env python

"""

	Adaptive Quadrature

"""

__author__  = "LJ Brown"
__file__ = "AdaptiveQuadrature.py"

import copy
import math
import json
import random
from itertools import product

import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.models import model_from_json

from LobattoQuadrature import LobattoQuadrature
from GaussLegendreQuadrature import GaussLegendreQuadrature
from GaussKronrodQuadrature import G7K15Quadrature

from local_error_methods import load_local_error_method

def magnitude(x):
	"""order of magnitude of x"""
	if x == 0.0:
		return int(math.log10(np.finfo(np.float32).eps))
	return int(math.log10(x))

class HashTable:

	def __init__(self):
		""" hash table for checking if nodes have already been evaluated f(xi). """

		# create empty hash table
		self.hash_table = {}

	def reuse_evaluation(self, f, x):

		# if node has already been evaluated
		# return value from hash table
		if x in self.hash_table.values():

			# additional function evaluations == 0
			return self.hash_table[x]

		# otherwise value by calling function
		# additional function evaluations == 1
		value = f(x)

		# add value to hash table
		self.hash_table[x] = value
		
		return value

	def get_nf(self):
		""" Return the total number of function calls performed """
		return len(self.hash_table)

class Node: 
	""" Binary tree node """
	
	def __init__(self, data_dict): 
		self.data_dict = data_dict

		# node children
		self.left = None
		self.right = None


def sum_leaf_entries(node, entry_key):

	if node is None: 
		return 0

	if ( node.left is None and node.right is None ): 
		return node.data_dict[entry_key]

	else: 
		return sum_leaf_entries(node.left, entry_key) + sum_leaf_entries(node.right, entry_key)

def count_leaf_nodes(node): 

	if node is None: 
		return 0 

	if(node.left is None and node.right is None): 
		return 1 

	else: 
		return count_leaf_nodes(node.left) + count_leaf_nodes(node.right) 


def global_error(root, true_error=None, output=False): 
	""" Gloabl error is the sum of the local errors in leaf nodes """

	g_error = sum_leaf_entries(root, 'local_error')

	if (true_error is not None) and (output is not False):

		error_in_approximation = magnitude(abs(g_error/true_error)) #magnitude(abs(true_error - g_error))
		print("Magnitude of error in error estimation: ", error_in_approximation)

	return g_error

def integral_approximation(root): 
	""" Integral approximation is the sum of the higher order method approximations """

	return sum_leaf_entries(root, 'In2')

def return_leaf_nodes(node):

	if node is not None: 

		if ( node.left is None and node.right is None ): 
			return [node]

		else: 
			return return_leaf_nodes(node.left) + return_leaf_nodes(node.right)

class Wrapper:

	"""
		Wrapper for quadrature methods from external source for testing against
	"""

	def __init__(self, quad_method, params_conversion_dict, return_conversion_dict, return_values_order_list, method_name):
		"""

			Wrapper class for external adaptive quadrature methods.


			example use:

				from scipy.integrate import quad

				cpd = 	{'epsabs' : 'tol', 'full_output' : '1',  'epsrel' : '0.0'}
				crd =  {'Iref' : '{\'Iapprox\' : return_vals[0]}', 'infodict' : '{\'nf\' : return_vals[2][\'neval\']}', 'Ierr' : '{}'}
				cro = 	['Iref', 'Ierr', 'infodict']
				method_name = "scipy quad"

				m = Wrapper(quad, cpd, crd, cro, method_name)

				t =float(10.0**(-4))
				others = {'Itrue' : 5}

				Iapprox, nf = m.adaptive_quad(f, a, b, tol=t, other_args=others)
		"""

		self.quad_method = quad_method
		self.params_conversion_dict = params_conversion_dict
		self.return_conversion_dict = return_conversion_dict
		self.return_values_order_list = return_values_order_list
		self.method_name = method_name

	def adaptive_quad(self, f, a, b, tol=1e-10, other_args={}):

		params = {}

		for k, v in self.params_conversion_dict.items():
			params[k] = eval(v)

		# run method
		return_vals = self.quad_method(f, a, b, **params)

		Iapprox = None
		nf = None
		return_dict = {}
		for rv_key in self.return_values_order_list:

			return_dict = { **return_dict, **eval(self.return_conversion_dict[rv_key])}

		return [return_dict['Iapprox'], return_dict['nf']]


	def __str__(self):
		return self.method_name

def leaf_node_statistics(root, Itrue=None):
	"""
		true error -- |Iture - Iapprox|
		sigma_delta_Iapprox - standard deviation of differences between 2 methods approximations for each leaf node
		mu_delta_Iapprox - mean of differences between 2 methods approximations for each leaf node
		n - number of leaf nodes
	"""
	n = count_leaf_nodes(root)

	# calculate mu and sigma
	leaf_nodes = return_leaf_nodes(root)
	delta_Iapproxes = []
	for ln_i in leaf_nodes:
		delta_Iapprox_i = abs(ln_i.data_dict['In2'] - ln_i.data_dict['In1'])
		delta_Iapproxes += [delta_Iapprox_i]

	sigma_delta_Iapprox, mu_delta_Iapprox = np.std(delta_Iapproxes), np.mean(delta_Iapproxes)

	stats = {
		'n' : n,
		'sigma_delta_Iapprox' : sigma_delta_Iapprox,
		'mu_delta_Iapprox' : mu_delta_Iapprox
	}

	if Itrue is not None:
		true_error = abs(integral_approximation(root) - Itrue)
		stats['true_error'] = true_error
	
	return stats


class AdaptiveQuadrature:

	def __init__(self, low_order_method, high_order_method, local_error_method_id=None, min_h=1e-06, method_order_difference=None, variable_h=False):

		# method should contain a member with the O(h^q) order approximation, q, called order
		self.low_order_method = low_order_method
		self.high_order_method = high_order_method

		self.min_h = min_h
		self.method_order_difference = method_order_difference
		self.variable_h = variable_h

		# pass local error method
		if local_error_method_id is not None:
			self.local_error_method_id = local_error_method_id
			self.local_error_method = load_local_error_method(self.local_error_method_id)
		else:
			self.local_error_method = self.error_estimate
			self.local_error_method_id = '4.2.2'

	def create_root(self, a, b):

		# compute first approximation
		In1, In2, local_error = self.quad(a, b)

		data_dict = {
			'a' : a,
			'b' : b,
			'In1' : In1,
			'In2' : In2,
			'local_error' : local_error
		}

		self.root = Node(data_dict)

	# if tolerence is not met, subdivide interval in node with largest local error
	# return number of function evaluations and updated tree's root node
	def sub_divide(self):

		# allow decrease in self.min_h
		if self.variable_h:
				self.max_local_error = getattr(self, "max_local_error", None)
				if self.max_local_error is None:
					self.max_local_error = np.inf

				if self.tol/self.max_local_error >= 1e-03:
					self.min_h = self.min_h/2

		# find node with maximum local error
		leaf_nodes = return_leaf_nodes(self.root)

		# don't do this randomly!
		np.random.shuffle(leaf_nodes)

		divide = False
		for ln in leaf_nodes:
			if abs(ln.data_dict['b'] - ln.data_dict['a'])/2 >= self.min_h:
				max_error_node = ln
				divide = True

		if divide == False:
			if self.output:
				print("Tree Stopped Dividing.")
			return False

		self.max_local_error = max_error_node.data_dict['local_error']

		for ln in leaf_nodes:

			# ensure the subinterval is not too small
			if abs(ln.data_dict['b'] - ln.data_dict['a'])/2 >= self.min_h:

				if ln.data_dict['local_error'] >= max_error_node.data_dict['local_error']:
					max_error_node = ln

		# create children by splitting the interval in two
		a, b = max_error_node.data_dict['a'], max_error_node.data_dict['b']
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

		max_error_node.left = Node(data_dict_l)
		max_error_node.right = Node(data_dict_r)

		# tree is still dividing
		return True

	def error_estimate(self, h, In1, In2):

		# chosen by heuristics
		# In1 is lower order methods estimation, In2 is higher order methods estimation
		d = abs(In2 - In1)

		# if method_order_difference memeber is defined use that
		method_order_difference = getattr(self, "method_order_difference", None)
		if method_order_difference is None:
			method_order_difference = self.high_order_method.order - self.low_order_method.order

		q = abs(method_order_difference)

		if h < 1:
			return d * h**q

		return d / h * 2

	def _error_estimate(self, h, In1, In2):
		""" call error estimate method. """
		return self.local_error_method(h, In1, In2)

	def quad(self, a, b):
		""" 

		Usage: 
			[In1, In2, err] = self.quad(f, a, b) 

		In1            -- Intergral approximation from low order method
		In2            -- Intergral approximation from high order method
		err 		   -- approximation of error over interval [a,b]

		"""

		ws1, xs1 = self.low_order_method.get_weights_and_nodes(a, b)
		ws2, xs2 = self.high_order_method.get_weights_and_nodes(a, b)

		# use hash table to retreive any previously computed values
		# to  minimize number of function calls -- nf -- hash_table.get_nf()
		fxs1 = []
		fxs2 = []
		for x in xs1:
			fx = self.hash_table.reuse_evaluation(self.f, x)
			fxs1.append(fx)

		for x in xs2:
			fx = self.hash_table.reuse_evaluation(self.f, x)
			fxs2.append(fx)

		# calculate integral approximations, reusing all function evaluations
		In1 = sum([w*fx for w,fx in zip(ws1,fxs1)]) 
		In2 = sum([w*fx for w,fx in zip(ws2,fxs2)])

		#
		# error approximation
		# 

		err = self._error_estimate(abs(b-a), In1, In2)

		return [In1, In2, err]


	def adaptive_quad(self, f, a, b, tol=1e-10, other_args={}):
		""" Wrapper method """
		return self.adaptive_quadrature(f, a, b, tol=tol, **other_args)


	def adaptive_quadrature(self, f, a, b, tol=1e-10, maxit=1000, output=False, Itrue=None, stats_file=None, nn_predict=None):

		self.f = f
		self.tol = tol
		self.output = output

		"""
		local_error_method=None, local_error_method_id=None,
		# pass local error method
		if local_error_method is not None:
			self.local_error_method = local_error_method
			self.local_error_method_id = local_error_method_id
		else:
			self.local_error_method = self.error_estimate
			self.local_error_method_id = '4.2.2'
		"""

		# for testing purposes include Itrue
		self.Itrue = Itrue

		# for testing purposes include stats file
		self.stats_file = stats_file

		# for testing purposes include nn error estimate prediction files
		self.nn_ee_files = nn_predict
		if self.nn_ee_files is not None:
			self.load_nn_error_estimate_model()

		# check inputs
		if (b < a):
			raise ValueError('adaptive_quadrature error: b < a!')

		# create empty hash table
		self.hash_table = HashTable()

		# create root node of tree
		self.create_root(a, b)

		for i in range(maxit):

			# exit if tolerence is met
			# include Itrue for testing error approximation
			true_error = None
			if Itrue is not None:
				true_error= abs(self.Itrue-integral_approximation(self.root))

			# write to stats file if Itrue and stats_file provided
			if self.Itrue is not None and self.stats_file is not None:
				self.append_stats_file()

			if nn_predict is not None:
				g_err = self.nn_global_error_estimate()
				if output and i%75 == 0 and true_error is not None:
					print("\ntrue error: %s,   predicted error: %s,   loss: %s" % (true_error, g_err, abs(true_error)-abs(g_err)))
					print("magnitude differences: %s" % (magnitude(true_error)-magnitude(g_err)))
			else:
				g_err = global_error(self.root, true_error=true_error, output=output)

			if magnitude(g_err) < magnitude(self.tol):

				if self.output:
					print("tolerence threshold met.")
					print("number of subintervals used: %s" %  count_leaf_nodes(self.root))

				return [integral_approximation(self.root), self.hash_table.get_nf()]

			# otherwise subdivide and try again
			err = abs(g_err - tol)
			still_dividing = self.sub_divide()

			# tree stopped dividing
			if not still_dividing:

				if self.output:
					print("tree stopped dividing")
					print("number of subintervals used: %s" %  count_leaf_nodes(self.root))

				return [integral_approximation(self.root), self.hash_table.get_nf()]
			
		if self.output:
			print("failure to meet tolernce")
			print("number of subintervals used: %s" %  count_leaf_nodes(self.root))

		return [integral_approximation(self.root), self.hash_table.get_nf()]

	def write_method(self, outfile):

		# assumes low and high order methods are of the same type
		method_name = self.low_order_method.get_method_name()

		low_order = self.low_order_method.order
		high_order = self.high_order_method.order

		# if method_order_difference memeber is defined get that
		method_order_difference = getattr(self, "method_order_difference", None)

		min_h = self.min_h
		varaible_h = self.variable_h

		method_params = {
			'low_order' : low_order,
			'high_order' : high_order,
			'method_order_difference' : method_order_difference, 
			'min_h' : min_h, 
			'variable_h' : varaible_h,
			'local_error_method_id' : self.local_error_method_id
		}

		method_data = {
			'method_name' : method_name,
			'method_params' : method_params
		}

		with open(outfile, 'w') as of:
			json.dump(method_data, of)

	def short_string(self):
		return self.low_order_method.short_string() + '-' + self.high_order_method.short_string()

	def __str__(self):			
		return '(' + str(self.low_order_method) + ', ' + str(self.high_order_method) + ')'


	@staticmethod
	def load_method(infile):

		with open(infile) as json_file:  
			method_data = json.load(json_file)

		method_name = method_data['method_name']
		method_params = method_data['method_params']

		return AdaptiveQuadrature.create_method(method_name, method_params)

	@staticmethod
	def create_method(method_name, params):

		"""
			example:
				m = create_method("LobattoQuadrature", params)
		"""

		if method_name == "LobattoQuadrature":
			low_order_method = LobattoQuadrature( params['low_order'] )
			high_order_method = LobattoQuadrature( params['high_order'] )

		elif method_name == "GaussLegendreQuadrature":
			low_order_method = GaussLegendreQuadrature( params['low_order'] )
			high_order_method = GaussLegendreQuadrature( params['high_order'] )

		elif method_name == "G7K15Quadrature":
			G7K15 = G7K15Quadrature()
			low_order_method = G7K15.get_G7()
			high_order_method = G7K15.get_K15()

		else: 
			raise("Error")

		# delete order params
		params_copy = copy.copy(params)
		del params_copy['low_order']
		del params_copy['high_order']

		# create method with specified hyper-parameters
		m = AdaptiveQuadrature(low_order_method, high_order_method, **params_copy)

		return m

	def append_stats_file(self):
		stats = leaf_node_statistics(self.root, self.Itrue)

		with open(self.stats_file, 'a') as f:
			json.dump(stats, f)
			f.write('\n')

	def nn_global_error_estimate(self): 
		stats = leaf_node_statistics(self.root)
		return self.estimate_error(stats)

	def load_nn_error_estimate_model(self):

		# load json and create model
		json_file = open(self.nn_ee_files['model_file'], 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)

		# load weights into new model
		loaded_model.load_weights(self.nn_ee_files['model_weights_file'])

		# evaluate loaded model on test data
		loaded_model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae']) 

		self.nn_model = loaded_model

	def estimate_error(self, stats_dict): 
		x = format_stats(stats_dict)

		# make prediction
		global_error_prediction = revert_prediction(self.nn_model.predict(x)[0][0])

		return global_error_prediction

def abslog10(x):
	"""abs log10 of x"""
	if x == 0.0:
		return abs(math.log10(np.finfo(np.float32).eps))
	return abs(math.log10(x))

def revert_prediction(y_hat):
	y_hat = -y_hat
	return 10**y_hat

def format_stats(stats_dict):

	x = np.empty(shape=(3,))
	x[0] = stats_dict['n']
	x[1] = abslog10(stats_dict['mu_delta_Iapprox']) 
	x[2] = abslog10(stats_dict['sigma_delta_Iapprox'])

	return x.reshape(1,3)