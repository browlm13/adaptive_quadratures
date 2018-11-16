#!/usr/bin/env python

"""

	local_error_methods

"""

__author__  = "LJ Brown"
__file__ = "local_error_methods.py"

import copy
import math
import json
import random
from itertools import product

import numpy as np
import glob

ERROR_METHODS_FILE_TEMPLATE = "error_methods/%s.json"

def get_method_file_ID(method_file_name):
	method_file_name = method_file_name.split('/')[-1]
	method_file_name = method_file_name.split('.')[0]
	method_file_name = method_file_name.replace("-", ".", 3)
	return str(method_file_name)

def get_method_file_ending(ID):
	method_file_ending = ID.replace(".", "-", 3)
	return str(method_file_ending)

def load_local_error_method(ID):
		global ERROR_METHODS_FILE_TEMPLATE

		# load method from generated methods folder
		method_file_name = ERROR_METHODS_FILE_TEMPLATE % get_method_file_ending(ID)

		print(method_file_name)

		with open(method_file_name) as json_file:  
			method_data = json.load(json_file)

		return eval(eval(method_data)['f'])