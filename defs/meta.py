# meta classifier
from common_defs import *

models = ( 'xgb', 'gb', 'rf', 'xt', 'sgd', 'polylearn_fm', 'polylearn_pn', 'keras_mlp' )

# import all the functions
for m in models:
	exec( "from defs.{} import get_params as get_params_{}" ).format( m, m )
	exec( "from defs.{} import try_params as try_params_{}" ).format( m, m )

space = { 'model': hp.choice( 'model', models ) }

def get_params():
	params = sample( space )
	m = params['model']
	m_params = eval( "get_params_{}()".format( m ))
	params.update( m_params )
	return params

def try_params( n_iterations, params ):
	params_ = dict( params )
	m = params_.pop( 'model' )
	print m
	
	return eval( "try_params_{}( n_iterations, params_ )".format( m ))
			 
	