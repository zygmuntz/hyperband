"""
load your data, which should be a dictionary with
x_train, y_train, x_test, y_test Numpy arrays
defs files import data from here (from load_data import data)
"""

# this particular example loads data from a pickle file

import cPickle as pickle

data_file = 'data/classification.pkl'

print "loading..."

with open( data_file, 'rb' ) as f:
	data = pickle.load( f )

"""
data is a dict containing numpy arrays: 
{ 'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test }
"""
