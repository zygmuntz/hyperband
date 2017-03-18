"function (and parameter space) definitions for hyperband"
"regression with Keras (multilayer perceptron)"

from common_defs import *

# a dict with x_train, y_train, x_test, y_test
from load_data_for_regression import data

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization as BatchNorm
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import *

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

#

# TODO: advanced activations - 'leakyrelu', 'prelu', 'elu', 'thresholdedrelu', 'srelu' 


max_layers = 5
max_layer_size = 100

space = {
	'scaler': hp.choice( 's', 
		( None, 'StandardScaler', 'RobustScaler', 'MinMaxScaler', 'MaxAbsScaler' )),
	'n_layers': hp.quniform( 'ls', 1, max_layers, 1 ),
	#'layer_size': hp.quniform( 'ls', 5, 100, 1 ),
	#'activation': hp.choice( 'a', ( 'relu', 'sigmoid', 'tanh' )),	
	'init': hp.choice( 'i', ( 'uniform', 'normal', 'glorot_uniform', 
		'glorot_normal', 'he_uniform', 'he_normal' )),
	'batch_size': hp.choice( 'bs', ( 16, 32, 64, 128, 256 )),
	'shuffle': hp.choice( 'sh', ( False, True )),
	'loss': hp.choice( 'l', ( 'mean_absolute_error', 'mean_squared_error' )),
	'optimizer': hp.choice( 'o', ( 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax' ))		
}

# for each hidden layer, we choose size, activation and extras individually
for i in range( 1, max_layers + 1 ):
	space[ 'layer_{}_size'.format( i )] = hp.quniform( 'ls{}'.format( i ), 
		2, max_layer_size, 1 )
	space[ 'layer_{}_activation'.format( i )] = hp.choice( 'a{}'.format( i ), 
		( 'relu', 'sigmoid', 'tanh' ))
	space[ 'layer_{}_extras'.format( i )] = hp.choice( 'e{}'.format( i ), ( 
		{ 'name': 'dropout', 'rate': hp.uniform( 'd{}'.format( i ), 0.1, 0.5 )}, 
		{ 'name': 'batchnorm' },
		{ 'name': None } ))	
	
def get_params():

	params = sample( space )
	return handle_integers( params )



#

# print hidden layers config in readable way
def print_layers( params ):
	for i in range( 1, params['n_layers'] + 1 ):
		print "layer {} | size: {:>3} | activation: {:<7} | extras: {}".format( i,
			params['layer_{}_size'.format( i )], 
			params['layer_{}_activation'.format( i )],
			params['layer_{}_extras'.format( i )]['name'] ),
		if params['layer_{}_extras'.format( i )]['name'] == 'dropout':
			print "- rate: {:.1%}".format( params['layer_{}_extras'.format( i )]['rate'] ),
		print

def print_params( params ):
	pprint({ k: v for k, v in params.items() if not k.startswith( 'layer_' )})
	print_layers( params )
	print	

def try_params( n_iterations, params ):
	
	print "iterations:", n_iterations
	print_params( params )
	
	y_train = data['y_train']
	y_test = data['y_test']
		
	if params['scaler']:
		scaler = eval( "{}()".format( params['scaler'] ))
		x_train_ = scaler.fit_transform( data['x_train'].astype( float ))
		x_test_ = scaler.transform( data['x_test'].astype( float ))
	else:
		x_train_ = data['x_train']
		x_test_ = data['x_test']
		
	input_dim = x_train_.shape[1]

	model = Sequential()
	model.add( Dense( params['layer_1_size'], init = params['init'], 
		activation = params['layer_1_activation'], input_dim = input_dim ))
	
	for i in range( int( params['n_layers'] ) - 1 ):
		
		extras = 'layer_{}_extras'.format( i + 1 )
		
		if params[extras]['name'] == 'dropout':
			model.add( Dropout( params[extras]['rate'] ))
		elif params[extras]['name'] == 'batchnorm':
			model.add( BatchNorm())
			
		model.add( Dense( params['layer_{}_size'.format( i + 2 )], init = params['init'], 
			activation = params['layer_{}_activation'.format( i + 2 )]))
		   
	model.add( Dense( 1, init = params['init'], activation = 'linear' ))

	model.compile( optimizer = params['optimizer'], loss = params['loss'] )
	
	#print model.summary()

	#

	validation_data = ( x_test_, y_test )

	early_stopping = EarlyStopping( monitor = 'val_loss', patience = 5, verbose = 0 )
	
	history = model.fit( x_train_, y_train,
		nb_epoch = int( round( n_iterations )),
		batch_size = params['batch_size'], 
		shuffle = params['shuffle'], 
		validation_data = validation_data, 
		callbacks = [ early_stopping ])	
	
	#
	
	p = model.predict( x_train_, batch_size = params['batch_size'] )

	mse = MSE( y_train, p )
	rmse = sqrt( mse )
	mae = MAE( y_train, p )


	print "\n# training | RMSE: {:.4f}, MAE: {:.4f}".format( rmse, mae )

	#

	p = model.predict( x_test_, batch_size = params['batch_size'] )
	
	mse = MSE( y_test, p )
	rmse = sqrt( mse )
	mae = MAE( y_test, p )

	print "# testing  | RMSE: {:.4f}, MAE: {:.4f}".format( rmse, mae )	
	
	return { 'loss': rmse, 'rmse': rmse, 'mae': mae, 'early_stop': model.stop_training }


