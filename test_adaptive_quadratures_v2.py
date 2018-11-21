#!/usr/bin/env python3

"""
		Tests performance of different adaptive quadrature methods
"""

__author__  = "LJ Brown"
__file__ = "test_adaptive_quadrature_v2.py"

# imports

# external
from numpy import *
from pylab import *
import glob
import pandas as pd 
import matplotlib.pyplot as plt

# dir
from test_integrals import test_integrals
from AdaptiveQuadrature import *

#
#
#  Import and wrap adaptive quadrature methods to test...
#
#

#
# reference method -- scipy quad
#

from scipy.integrate import quad

# wrap
cpd4 = 	{'epsabs' : 'tol', 'full_output' : '1',  'epsrel' : '0.0'}
crd4 =  {'Iref' : '{\'Iapprox\' : return_vals[0]}', 'infodict' : '{\'nf\' : return_vals[2][\'neval\']}', 'Ierr' : '{}'}
cro4 = 	['Iref', 'Ierr', 'infodict']
method_name1 = "scipy quad"
reference_method = Wrapper(quad, cpd4, crd4, cro4, method_name1)

#
# adaptive lobatto
#

from adaptive_lobatto import AdaptiveLobatto1213
adapt_lobatto = AdaptiveLobatto1213()

#
# Dr. Reynolds adaptive lobatto
#

from adaptive_lobatto_DRR import adaptive_lobatto as adapt_lobatto_DRR

# wrap
cpdr = 	{'tol' : 'tol'}
crdr =  {'Iapprox' : '{\'Iapprox\' : return_vals[0]}', 'nf' : '{\'nf\' : return_vals[1]}'}
cror = 	['Iapprox', 'nf']
method_name3 = "Dr.R"
adapt_lobatto_drr = Wrapper(adapt_lobatto_DRR, cpdr, crdr, cror, method_name3)

#
# Methods to test
#

test_methods = [reference_method, adapt_lobatto, adapt_lobatto_drr]

#
# Test Settings
#

# other settings
CONSOLE_OUTPUT = True
PLOT = True
min_tol, max_tol = -4, -10
num_tols = abs(max_tol - min_tol) + 1
REFERENCE_METHOD = method_name3 		# set reference method
additional_settings_parameters = {'output' : False, 'maxit' : 100}

#
#  helper methods
#

def geometric_mean(v):
	""" returns geometric mean of numpy vector """
	return np.prod(np.power(v,1/v.size))

def plot_error_v_tol(df, plot_title=None):
	error_tol_df = df[['error', 'tol', 'method_name']]
	error_tol_df.set_index('tol', inplace=True)
	error_tol_df.groupby('method_name')['error'].plot(legend=True, loglog=True, title=plot_title)

	plt.show()


def plot_nf_v_tol(df,tols, plot_title=None):
	nrows = 2
	ncols = math.ceil(len(tols)/2)
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

	if plot_title is not None:
		fig.suptitle(plot_title, fontsize=14)

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

def run_integral_trials(min_tol, max_tol, num_tols, integral_name, test_integrals, test_methods, additional_settings_parameters={}, CONSOLE_OUTPUT=False, PLOT=False):
	"""
		Run tests on test integral

		for m, method in enumerate(test_methods):
			for t, tol in enumerate(tols) :
				err[m,t] = |Itrue - Iapprox|
				rat[m,t] = err[m,t]/tols[t]
				wk[m,t] = nf

		# matrices
		returns err, rat, wk
	"""

	# set the integration interval, integrand function and parameters
	f, a, b, Itrue = test_integrals[integral_name]['f'], test_integrals[integral_name]['a'], test_integrals[integral_name]['b'], test_integrals[integral_name]['Itrue'] 

	# add Itrue to additional settings parameters
	additional_settings_parameters['Itrue'] = Itrue

	# set up tolerances
	tols = np.logspace(min_tol, max_tol, num=num_tols)

	# allocate space for integral statistic vectors 
	nrows, ncols = len(test_methods),num_tols
	err = np.zeros(shape=(nrows, ncols), dtype=float)
	rat = np.zeros(shape=(nrows, ncols), dtype=float)
	wk = np.zeros(shape=(nrows, ncols), dtype=int)

	# run trials
	trial_logs = []
	for t, tol in enumerate(tols) :

		for m, method in enumerate(test_methods):

			# perform approximation (or at least try)
			try:
				Iapprox, nf = method.adaptive_quad(f, a, b, tol=tol, other_args=additional_settings_parameters)
			except:
				exit

			error = abs(Itrue - Iapprox)

			# store statistics in trial matrices
			err[m,t] = error
			rat[m,t] = error/tol
			wk[m,t] = nf

			# update trial logs
			log = {
				'method_name' : str(method),
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

	#
	# output
	#

	if CONSOLE_OUTPUT or PLOT:
		trials_df = pd.DataFrame(trial_logs)

	#
	# plot results
	#

	if PLOT:
		plot_error_v_tol(trials_df, plot_title=integral_name)
		plot_nf_v_tol(trials_df, tols, plot_title=integral_name)

	# return test integral statistics
	return err, rat, wk

def run_trials(min_tol, max_tol, num_tols, test_integrals, test_methods, additional_settings_parameters={}, CONSOLE_OUTPUT=False, PLOT=False):

	# initalize statistics matrices
	nrows, ncols = len(test_methods), num_tols*len(test_integrals)
	Err = np.zeros(shape=(nrows, ncols), dtype=float)
	Rat = np.zeros(shape=(nrows, ncols), dtype=float)
	Wk = np.zeros(shape=(nrows, ncols), dtype=int)

	# run trials for each test integral
	test_integral_names = [k for k in test_integrals.keys()]
	for i, ti_name in enumerate(test_integral_names):

		# run
		err, rat, wk = run_integral_trials(min_tol, max_tol, num_tols, ti_name, test_integrals, test_methods, additional_settings_parameters=additional_settings_parameters, CONSOLE_OUTPUT=CONSOLE_OUTPUT, PLOT=PLOT)

		# store statistics
		c1, c2 = num_tols*(i),num_tols*(i+1)
		Err[:,c1:c2] = err
		Rat[:,c1:c2] = rat
		Wk[:,c1:c2] = wk

	#display output
	if CONSOLE_OUTPUT:

		method_stats = pd.DataFrame(columns=['method', 'max err', 'combined err', 'combined nf'])

		row2name = {i:str(m) for i,m in zip(range(len(test_methods)),test_methods)}

		for r in range(len(test_methods)):
			rat = Rat[r,:]
			wk = Wk[r,:]

			# compile statistics
			rat_tot = np.mean(rat) #geometric_mean(rat)
			rat_max = np.max(rat)
			wk_tot  = geometric_mean(wk)

			method_stats.loc[r] = [row2name[r], rat_max, rat_tot, wk_tot]

		# print output
		print(method_stats)


# run trials
if __name__ == '__main__':
	run_trials(min_tol, max_tol, num_tols, test_integrals, test_methods, additional_settings_parameters=additional_settings_parameters, CONSOLE_OUTPUT=CONSOLE_OUTPUT, PLOT=PLOT)
