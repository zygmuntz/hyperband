"function (and parameter space) definitions for hyperband"
"regression with polylearn polynomial networks"

from common_defs import *

# a dict with x_train, y_train, x_test, y_test
from load_data_for_regression import data

from polylearn import PolynomialNetworkRegressor as PN
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

#

iters_per_iteration = 1

space = {
	'scaler': hp.choice( 's', 
		( None, 'StandardScaler', 'RobustScaler', 'MinMaxScaler', 'MaxAbsScaler' )),	
	
	'degree': hp.quniform( 'd', 2, 3, 1 ),
	'n_components': hp.quniform( 'c', 1, 10, 1 ),
	#'alpha': hp.uniform( 'a', 1e-10, 2 ),
	'beta': hp.uniform( 'b', 1e-10, 2 ), 
	'fit_lower': hp.choice( 'flo', ( 'augment', None )),	# 'explicit'
	#'fit_linear': hp.choice( 'fln', ( False, True )),
	#'init_lambdas': hp.choice( 'il', ( 'ones', 'random_signs' ))
}

def get_params():

	params = sample( space )
	return handle_integers( params )

#

def try_params( n_iterations, params ):
	
	max_iter = int( round( n_iterations * iters_per_iteration ))
	print "max_iter:", max_iter
	pprint( params )
	
	if params['scaler']:
		scaler = eval( "{}()".format( params['scaler'] ))
		x_train_ = scaler.fit_transform( data['x_train'].astype( np.float64 ))
		x_test_ = scaler.transform( data['x_test'].astype( np.float64 ))
	else:
		x_train_ = data['x_train'].astype( np.float64 )
		x_test_ = data['x_test'].astype( np.float64 )
		
	y_train_ = data['y_train'].copy()
	y_test_ = data['y_test'].copy()
	
	local_data = { 'x_train': x_train_, 'y_train': y_train_, 
		  'x_test': x_test_ , 'y_test': y_test_ }
	
	#
	
	params_ = dict( params )
	params_.pop( 'scaler' )	
	
	clf = PN( max_iter = max_iter, verbose = True, **params_ )
	return train_and_eval_sklearn_regressor( clf, local_data )
