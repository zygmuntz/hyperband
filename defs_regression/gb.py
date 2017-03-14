"function (and parameter space) definitions for hyperband"
"regression with gradient boosting"

from common_defs import *

# a dict with x_train, y_train, x_test, y_test
from load_data_for_regression import data

from sklearn.ensemble import GradientBoostingRegressor as GB

#

trees_per_iteration = 5

# subsample is good for showing out-of-bag errors 
# when fitting in verbose mode, and probably not much else
space = {
	'loss': hp.choice( 'l', ( 'ls', 'lad', 'huber', 'quantile' )),
	'alpha': hp.uniform( 'a', 0.5, 0.95 ),	# for huber & quantile losses
	'criterion': hp.choice( 'c', ( 'friedman_mse', 'mse', 'mae' )),	# sklearn 0.18
	
	'learning_rate': hp.uniform( 'lr', 0.01, 0.2 ),
	'subsample': hp.uniform( 'ss', 0.8, 1.0 ),
	'max_depth': hp.quniform( 'md', 2, 10, 1 ),
	'max_features': hp.choice( 'mf', ( 'sqrt', 'log2', None )),
	'min_samples_leaf': hp.quniform( 'mss', 1, 10, 1 ),
	'min_samples_split': hp.quniform( 'mss', 2, 20, 1 )
}

def get_params():

	params = sample( space )
	return handle_integers( params )

#

def try_params( n_iterations, params ):

	n_estimators = int( round( n_iterations * trees_per_iteration ))
	print "n_estimators:", n_estimators
	pprint( params )

	clf = GB( n_estimators = n_estimators, verbose = 0, **params )

	return train_and_eval_sklearn_regressor( clf, data )

	