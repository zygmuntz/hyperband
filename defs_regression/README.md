functions and search space definitions for regression
=====================================================

This directory contains implementations of `get_params()`, `try_params()` and search space for various regressors (the same as classifiers).

	meta.py - a meta regressor

	gb.py - gradient boosting
	rf.py - random forest
	rf_xt.py - random forest and extra trees together
	sgd.py - a linear SGD regressor
	xt.py - extremely randomized trees	
	
	xgb.py - XGBoost regressor
	
	keras_mlp.py - a Keras MLP
	
	polylearn_fm.py - polylearn factorization machine
	polylearn_fm_pn.py - FM + PN
	polylearn_pn.py - polylearn polynomial network

	
Note that you can use these definitions with [hyperopt](https://github.com/hyperopt/hyperopt).	