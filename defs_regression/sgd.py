"function (and parameter space) definitions for hyperband"
"regression with linear SGD regressor"

from common_defs import *

# a dict with x_train, y_train, x_test, y_test
from load_data_for_regression import data

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import SGDRegressor as SGD

#

space = {
	'scaler': hp.choice( 's', 
		( None, 'StandardScaler', 'RobustScaler', 'MinMaxScaler', 'MaxAbsScaler' )),	
	'loss': hp.choice( 'l', ( 'squared_loss', 'huber', 
		'epsilon_insensitive', 'squared_epsilon_insensitive' )),	
	'epsilon': hp.uniform( 'e', 0.001, 0.2 ),
	'penalty': hp.choice( 'p', ( 'none', 'l1', 'l2', 'elasticnet' )),
	'alpha': hp.loguniform( 'a', log( 1e-10 ), log( 1 )),
	'l1_ratio': hp.uniform( 'l1r', 0, 1 ),
	'fit_intercept': hp.choice( 'i', (True, False )),
	'shuffle': hp.choice( 'sh', ( True, False )),
	'learning_rate': hp.choice( 'lr', ( 'constant', 'optimal', 'invscaling' )),
	'eta0': hp.loguniform( 'eta', log( 1e-10 ), log( 1 )),
	'power_t': hp.uniform( 'pt', 0.5, 0.99 ),
}

def get_params():

	params = sample( space )
	return handle_integers( params )

#

def try_params( n_iterations, params ):
	
	n_iterations = int( round( n_iterations ))
	print "n_iterations:", n_iterations
	pprint( params )
	
	if params['scaler']:
		scaler = eval( "{}()".format( params['scaler'] ))
		x_train_ = scaler.fit_transform( data['x_train'].astype( float ))
		x_test_ = scaler.transform( data['x_test'].astype( float ))
		
		local_data = { 'x_train': x_train_, 'y_train': data['y_train'], 
		  'x_test': x_test_, 'y_test': data['y_test'] }
	else:
		local_data = data
	
	# we need a copy because at the next small round the best params will be re-used
	params_ = dict( params )
	params_.pop( 'scaler' )
	
	clf = SGD( n_iter = n_iterations, **params_ )
	
	return train_and_eval_sklearn_regressor( clf, local_data )