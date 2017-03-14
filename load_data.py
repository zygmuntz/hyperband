"load data, which should be a dictionary with"
"x_train, y_train, x_test, y_test - numpy arrays"
"defs files import data from here"

import cPickle as pickle

data_file = 'data/classification.pkl'

print "loading..."

with open( data_file, 'rb' ) as f:
	data = pickle.load( f )



