#!/usr/bin/env python

"load pickled results, show the best"

import sys
import pickle

from pprint import pprint

try:
	input_file = sys.argv[1]
except IndexError:
	print "Usage: python show_results.py <results.pkl> [<number of results to show>]\n"
	raise SystemExit

try:
	results_to_show = int( sys.argv[2] )
except IndexError:
	results_to_show = 5

with open( input_file, 'rb' ) as i_f:
	results = pickle.load( i_f )

for r in sorted( results, key = lambda x: x['loss'] )[:results_to_show]:
	print "loss: {:.2%} | {} seconds | {:.1f} iterations | run {} ".format( 
		r['loss'], r['seconds'], r['iterations'], r['counter'] )
	pprint( r['params'] )
	print