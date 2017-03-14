"function (and parameter space) definitions for hyperband"
"regression with extremely randomized trees"

from common_defs import *

# a dict with x_train, y_train, x_test, y_test
from load_data_for_regression import data

from sklearn.ensemble import ExtraTreesRegressor as XT

#

trees_per_iteration = 5

space = {
	'criterion': hp.choice( 'c', ( 'mse', 'mae' )),		# sklearn 0.18
	'bootstrap': hp.choice( 'b', ( True, False )),
	'max_depth': hp.quniform( 'md', 2, 10, 1 ),
	'max_features': hp.choice( 'mf', ( 'sqrt', 'log2', None )),
	'min_samples_split': hp.quniform( 'msp', 2, 20, 1 ),
	'min_samples_leaf': hp.quniform( 'msl', 1, 10, 1 ),
}

def get_params():

	params = sample( space )
	return handle_integers( params )

#

def try_params( n_iterations, params ):
	
	n_estimators = int( round( n_iterations * trees_per_iteration ))
	print "n_estimators:", n_estimators
	pprint( params )
	
	clf = XT( n_estimators = n_estimators, verbose = 0, n_jobs = -1, **params )
	return train_and_eval_sklearn_regressor( clf, data )



	