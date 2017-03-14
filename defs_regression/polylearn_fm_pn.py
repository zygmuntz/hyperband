"function (and parameter space) definitions for hyperband"
"regression with polylearn FM/PN"

from common_defs import *

# a dict with x_train, y_train, x_test, y_test
from load_data_for_regression import data

from polylearn import FactorizationMachineRegressor as FM
from polylearn import PolynomialNetworkRegressor as PN
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

#

iters_per_iteration = 1

space = {
	'scaler': hp.choice( 's', 
		( None, 'StandardScaler', 'RobustScaler', 'MinMaxScaler', 'MaxAbsScaler' )),	
	
	'classifier': hp.choice( 'c', (
		{ 'name': 'FM', 
		  'local_params': {
			'alpha': hp.uniform( 'a', 1e-10, 2 ),
			'fit_linear': hp.choice( 'fln', ( False, True )),
			'fit_lower': hp.choice( 'flo', ( 'augment', 'explicit', None )),
			'init_lambdas': hp.choice( 'il', ( 'ones', 'random_signs' ))
		}},
		{ 'name': 'PN', 
		  'local_params': {
			'fit_lower': hp.choice( 'flo', ( 'augment', None ))
		}}
	)),
	'degree': hp.quniform( 'd', 2, 3, 1 ),
	'n_components': hp.quniform( 'c', 1, 50, 1 ),
	'beta': hp.uniform( 'b', 1e-10, 2 ), 
	
}

def get_params():

	params = sample( space )
	return handle_integers( params )

#

def try_params( n_iterations, params ):
	
	max_iter = int( round( n_iterations * iters_per_iteration ))
	print "max_iter:", max_iter
	
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
	
	classifier = params['classifier']['name']
	local_params = params['classifier']['local_params']
	
	params_ = dict( params )
	params_.pop( 'classifier' )
	params_.update( local_params )
	
	print classifier
	pprint( params_ )	
	
	params_.pop( 'scaler' )	

	
	clf = eval( "{}( max_iter = max_iter, verbose = True, \
		**params_ )".format( classifier ))
	return train_and_eval_sklearn_regressor( clf, local_data )
