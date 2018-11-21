#!/usr/bin/env python3

"""

	File of test integrals and their true solutions.

"""
__author__  = "LJ Brown"
__file__ = "test_integrals.py"

# imports

# external
import numpy as np

#
# helper methods
#

def add_test_integral(test_integrals, integral_name, a, b, f, Itrue):
	""" add new test integral to test_integrals dictionary and return """
	test_integrals[integral_name] = {
		'a' : a,
		'b' : b,
		'f' : f,
		'Itrue' : Itrue
	}
	return test_integrals

#
# test integrals dictionary
#

test_integrals = {}

#
# test integral 1
#

integral_name1 = "I1"

a1 = -3.0
b1 = 3.0

def f1(x):
	return (np.cos(x**4) - np.exp(-x/3))

# true solution (from Mathematica)
I1 = -5.388189110199078390464788757549832333192362851501884776675107808988626164717563118491875769130202907

# add to dict
test_integrals = add_test_integral(test_integrals, integral_name1, a1, b1, f1, I1)

#
# test integral 2
#

integral_name2 = "I2"

a2 = -5.0
b2 = 5.0

def f2(x):
	return (np.cos(x**4) - np.exp(-x/3))

# true solution (from Mathematica)
I2 = -13.641321161632866996537541724410576129546923463595565446127805228232518667085483087327115590189547216

# add to dict
test_integrals = add_test_integral(test_integrals, integral_name2, a2, b2, f2, I2)

#
# test integral 3
#

integral_name3 = "I3"

a3 = 0.0
b3 = 2*np.pi

def f3(x):
	return np.cos(x)**2

I3 = np.pi

# add to dict
test_integrals = add_test_integral(test_integrals, integral_name3, a3, b3, f3, I3)

#
# test integral 4
#

integral_name4 = "I4"

a4 = 0.01
b4 = 10000

def f4(x):
    return (x * np.exp(-x**2) * np.sin(np.exp(-x**2)))

I4 = (np.cos(np.exp(-b4**2)) - np.cos(np.exp(-a4**2)))/2

# add to dict
test_integrals = add_test_integral(test_integrals, integral_name4, a4, b4, f4, I4)

#
# test integral 4
#

integral_name5 = "I5"

a5 = 20
b5 = 20.01

def f5(x):
    return (x**3 * np.sin(x**4))

I5 = (-np.cos(b5**4) + np.cos(a5**4))/4

# add to dict
test_integrals = add_test_integral(test_integrals, integral_name5, a5, b5, f5, I5)

#
# test integral 6
#

integral_name6 = "I6"

a6 = -10
b6 = -8

def f6(x):
    return (x**2 * np.cos(2 * x**3))

I6 = (np.sin(2*b6**3) - np.sin(2*a6**3))/6

# add to dict
test_integrals = add_test_integral(test_integrals, integral_name6, a6, b6, f6, I6)


