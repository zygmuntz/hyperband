"imports and definitions shared by various defs files"

import numpy as np

from math import log
from time import time
from pprint import pprint

from sklearn.metrics import roc_auc_score as AUC, log_loss, accuracy_score as accuracy

try:
	from hyperopt import hp
	from hyperopt.pyll.stochastic import sample
except ImportError:
	print "In order to achieve operational capability, this programme requires hyperopt to be installed (pip install hyperopt), unless you make get_params() use something else."
	
#	

# handle floats which should be integers
# works with flat params
def handle_integers( params ):

	new_params = {}
	for k, v in params.items():
		if type( v ) == float and int( v ) == v:
			new_params[k] = int( v )
		else:
			new_params[k] = v
	
	return new_params
	
	


def train_and_eval_sklearn_classifier( clf, data ):
	
	x_train = data['x_train']
	y_train = data['y_train']
	
	clf.fit( x_train, y_train )	
	
	return evaluate_classifier( clf, data )

def evaluate_classifier( clf, data ):

	x_train = data['x_train']
	y_train = data['y_train']

	x_test = data['x_test']
	y_test = data['y_test']
	
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