#!/usr/bin/env python3

"""
		Tests performance of different adaptive quadrature methods
"""

__author__  = "LJ Brown"
__file__ = "test_adaptive_quadrature.py"

# imports

# external
from numpy import *
from adaptive_trap import *
from scipy.integrate import quad
from pylab import *
import glob
import pandas as pd 
import matplotlib.pyplot as plt

# dir
from AdaptiveQuadrature import *

#
# reference method -- scipy quad
#

cpd4 = 	{'epsabs' : 'tol', 'full_output' : '1',  'epsrel' : '0.0'}
crd4 =  {'Iref' : '{\'Iapprox\' : return_vals[0]}', 'infodict' : '{\'nf\' : return_vals[2][\'neval\']}', 'Ierr' : '{}'}
cro4 = 	['Iref', 'Ierr', 'infodict']
method_name = "scipy quad"

reference_method = Wrapper(quad, cpd4, crd4, cro4, method_name)

# set as reference method
REFERENCE_METHOD = method_name

#
# reference method -- adaptive trap
# 

cpd5 = 	{'tol' : 'tol'}
crd5 =  {'Iapprox' : '{\'Iapprox\' : return_vals[0]}', 'nf' : '{\'nf\' : return_vals[1]}'}
cro5 = 	['Iapprox', 'nf']
method_name = "adaptive trap"

adaptive_trap = Wrapper(adaptive_trap, cpd5, crd5, cro5, method_name)

#
# Method 0-700
#

low_order_method_0_700 = LobattoQuadrature(12)
high_order_method_0_700 = LobattoQuadrature(13)

# Adaptive Quadrature Class
m_0_700 = AdaptiveQuadrature(low_order_method_0_700, high_order_method_0_700, method_order_difference=3.38, min_h=0.01, variable_h=True)

#
# Method 0-1000
#

low_order_method_0_1000 = GaussLegendreQuadrature(11)
high_order_method_0_1000 = GaussLegendreQuadrature(13)

# Adaptive Quadrature Class
m_0_1000 = AdaptiveQuadrature(low_order_method_0_1000, high_order_method_0_1000, method_order_difference=5, min_h=0.5, variable_h=True) 

#
# Method G7K15-2
#

G7K15 = G7K15Quadrature()
low_order_method_G7K15_2 = G7K15.get_G7()
high_order_method_G7K15_2 = G7K15.get_K15()

# Adaptive Quadrature Class
m_G7K15_2 = AdaptiveQuadrature(low_order_method_G7K15_2, high_order_method_G7K15_2, method_order_difference=15, min_h=0.01, variable_h=True) 

#
# adaptive lobatto
#
from adaptive_lobatto import AdaptiveLobatto1213
adapt_lobatto = AdaptiveLobatto1213()

#
# Methods to test
#

test_methods = [reference_method, m_0_700, m_0_1000, m_G7K15_2, adapt_lobatto]

#
# Test Settings
#


# set the integration interval, integrand function and parameters
a = -3.0
b = 3.0

def f(x):
	return (cos(x**4) - exp(-x/3))

# true solution (from Mathematica)
Itrue = -5.388189110199078390464788757549832333192362851501884776675107808988626164717563118491875769130202907

"""
a = -5.0
b = 5.0
def f(x):
	return (cos(x**4) - exp(-x/3))

Itrue = -13.641321161632866996537541724410576129546923463595565446127805228232518667085483087327115590189547216
"""
"""
a = 0.0
b = 2*np.pi
def f(x):
	return cos(x)**2
Itrue = np.pi
"""

# other settings
CONSOLE_OUTPUT = True
additional_settings_parameters = {'Itrue' : Itrue, 'output' : False, 'maxit' : 100}
min_tol, max_tol = 1, 16



def plot_error_v_tol(df):
	error_tol_df = df[['error', 'tol', 'method_name']]
	error_tol_df.set_index('tol', inplace=True)
	error_tol_df.groupby('method_name')['error'].plot(legend=True, loglog=True)

	plt.show()


def plot_nf_v_tol(df,tols):
	nrows = 2
	ncols = math.ceil(len(tols)/2)
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

	for i,tol in enumerate(tols):
		if i >= int(len(tols)/2): 
			j, i = 1, i - int(len(tols)/2)
		else: 
			j = 0

		tol_df = df.loc[df['tol'] == tol]
		nf_totals = tol_df.groupby('method_name')['nf'].sum()
		nf_totals = nf_totals[nf_totals.index != 'requirment']

		nf_totals.plot.bar(ax=axes[j,i])

		# set the x-axis label
		x = range(len(nf_totals))
		axes[j,i].set_xlabel("")

		# title
		title = "tol %s" % tol
		axes[j,i].set_title(title)

		# draw line from reference method nf value
		reference_threshold = nf_totals[nf_totals.index == REFERENCE_METHOD][0]

		above_threshold = np.maximum(nf_totals - reference_threshold, 0)
		below_threshold = np.minimum(nf_totals, reference_threshold)

		axes[j,i].bar(x, above_threshold, 0.5, color="r", bottom=below_threshold)

		# horizontal line indicating the reference method threshold nf value
		axes[j,i].plot([0., 4.5], [reference_threshold, reference_threshold], "k--")

	plt.subplots_adjust(bottom=0.15)
	plt.subplots_adjust(top=0.88)
	plt.subplots_adjust(left=0.15)
	plt.subplots_adjust(right=0.85)
	plt.subplots_adjust(wspace=0.85)
	plt.subplots_adjust(hspace=1.0)

	plt.show()

#
# Run Tests
#

# set up tolerances
ts = arange(min_tol,max_tol)
tols = [10.0**(-t) for t in ts]

trial_logs = []
for tol in tols:

	for m in test_methods:

		Iapprox, nf = m.adaptive_quad(f, a, b, tol=tol, other_args=additional_settings_parameters)
		error = abs(Itrue - Iapprox)

		log = {
			'method_name' : str(m),
			'Iapprox': Iapprox,
			'nf' : nf,
			'tol' : tol,
			'error' : error
		}

		trial_logs += [log]

	# Add tolerance requirment as own method
	log = {
			'method_name' : "requirment",
			'Iapprox': Itrue,
			'nf' : None,
			'tol' : tol,
			'error' : tol
	}
	trial_logs += [log]


trials_df = pd.DataFrame(trial_logs)

if CONSOLE_OUTPUT:
	print(trials_df[['nf', 'method_name']])


#
# plot results
#

plot_error_v_tol(trials_df)
plot_nf_v_tol(trials_df, tols)
