#!/usr/bin/env python

"bare-bones demonstration of using hyperband to tune sklearn GBT"

from hyperband import Hyperband
from defs.gb import get_params, try_params

hb = Hyperband( get_params, try_params )

# no actual tuning, doesn't call try_params()
# results = hb.run( dry_run = True )		

results = hb.run( skip_last = 1 )		# shorter run
results = hb.run()

