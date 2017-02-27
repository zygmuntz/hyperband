hyperband
=========

Code for tuning hyperparams with Hyperband, adapted from [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://people.eecs.berkeley.edu/~kjamieson/hyperband.html). Includes definitions for scikit-learn gradient boosting.

	defs_gb.py - functions and search space for gradient boosting
	hyperband.py - from hyperband import Hyperband
	load_data.py - defs_gb imports from this
	main.py - a complete example
	main_simple.py - a simple example	
	
Usage
-----

Edit `load_data.py` or `defs_gb.py` directly to provide your data. Then run `main.py`.

