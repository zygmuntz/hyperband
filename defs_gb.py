"function (and parameter space) definitions for hyperband"
"binary classification with gradient boosting"

import numpy as np

from math import log
from time import time
from pprint import pprint

from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.metrics import roc_auc_score as AUC, log_loss, accuracy_score as accuracy

try:
	from hyperopt import hp
	from hyperopt.pyll.stochastic import sample
except ImportError:
	print "In order to achieve operational capability, this programme requires hyperopt to be installed (pip install hyperopt)."

from load_data import x_train, y_train, x_test, y_test

#

trees_per_iteration = 5

# subsample is good for showing out-of-bag errors 
# when fitting in verbose mode, and probably not much else
space = {
	'learning_rate': hp.uniform( 'lr', 0.01, 0.2 ),
	'subsample': hp.uniform( 'ss', 0.8, 1.0 ),
	'max_depth': hp.quniform( 'md', 2, 10, 1 ),
	'max_features': hp.choice( 'mf', ( 'sqrt', 'log2', None )),
	'min_samples_leaf': hp.quniform( 'mss', 1, 10, 1 ),
	'min_samples_split': hp.quniform( 'mss', 2, 20, 1 )
}

#

def get_params():

	params = sample( space )
	
	# handle floats which should be integers
	new_params = {}
	for k, v in params.items():
		if type( v ) == float and int( v ) == v:
			new_params[k] = int( v )
		else:
			new_params[k] = v
			
	return new_params


def try_params( n_iterations, params ):

	n_estimators = int( round( n_iterations * trees_per_iteration ))
	print "n_estimators:", n_estimators
	pprint( params )

	clf = GB( n_estimators = n_estimators, verbose = 0, **params )
	clf.fit( x_train, y_train )	
	
	p = clf.predict_proba( x_train )[:,1]

	ll = log_loss( y_train, p )
	auc = AUC( y_train, p )
	acc = accuracy( y_train, np.round( p ))

	print "\n# training | log loss: {:.2%}, AUC: {:.2%}, accuracy: {:.2%}".format( ll, auc, acc )

	#

	p = clf.predict_proba( x_test )[:,1]

	ll = log_loss( y_test, p )
	auc = AUC( y_test, p )
	acc = accuracy( y_test, np.round( p ))

	print "# testing  | log loss: {:.2%}, AUC: {:.2%}, accuracy: {:.2%}".format( ll, auc, acc )	
	
	return { 'loss': ll, 'log_loss': ll, 'auc': auc }
	