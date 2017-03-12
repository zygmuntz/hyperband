functions and search space definitions
======================================

This directory contains implementations of `get_params()`, `try_params()` and search space for various classifiers.

	meta.py - a meta classifier

	gb.py - gradient boosting
	rf.py - random forest
	rf_xt.py - random forest and extra trees together
	sgd.py - a linear SGD classifier
	xt.py - extremely randomized trees	
	
	keras_mlp.py - a Keras MLP
	
	polylearn_fm.py - [polylearn](https://github.com/scikit-learn-contrib/polylearn/) factorization machine
	polylearn_fm_pn.py - FM + PN
	polylearn_pn.py - polylearn polynomial network

	
Note that you can use these definitions with [hyperopt](https://github.com/hyperopt/hyperopt).	