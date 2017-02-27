import cPickle as pickle

data_file = 'data/xy.pkl'

with open( data_file, 'rb' ) as f:
	data = pickle.load( f )

x_train = data['x_train']
y_train = data['y_train']

x_test = data['x_test']
y_test = data['y_test']

