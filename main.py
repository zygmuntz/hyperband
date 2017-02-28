#!/usr/bin/env python

"a more polished example of using hyperband to tune GBT"
"includes displaying best results and saving to a file"

import sys
import cPickle as pickle
from pprint import pprint

from hyperband import Hyperband
#from defs_gb import get_params, try_params
#from defs_rf import get_params, try_params
#from defs_xt import get_params, try_params
from defs_rf_xt import get_params, try_params

try:
	output_file = sys.argv[1]
	if not output_file.endswith( '.pkl' ):
		output_file += '.pkl'	
except IndexError:
	output_file = 'results.pkl'
	
print "will save results to", output_file	

#

hb = Hyperband( get_params, try_params )
results = hb.run( skip_last = 1 )

print "{} total, best:\n".format( len( results ))

for r in sorted( results, key = lambda x: x['loss'] )[:5]:
	print "loss: {:.2%} | {} seconds | {:.1f} iterations | run {} ".format( 
		r['loss'], r['seconds'], r['iterations'], r['counter'] )
	pprint( r['params'] )
	print


with open( output_file, 'wb' ) as f:
	pickle.dump( results, f )