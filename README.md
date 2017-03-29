hyperband
=========

Code for tuning hyperparams with Hyperband, adapted from [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://people.eecs.berkeley.edu/~kjamieson/hyperband.html). 

	defs/ - functions and search space definitions for various classifiers
	defs_regression/ - the same for regression models
	common_defs.py - imports and definitions shared by defs files
	hyperband.py - from hyperband import Hyperband
	
	load_data.py - classification defs import data from this file
	load_data_regression.py - regression defs import data from this file
	
	main.py - a complete example for classification
	main_regression.py - the same, for regression
	main_simple.py - a simple, bare-bones, example	

The goal is to provide a fully functional implementation of Hyperband, as well as a number of ready to use functions for a number of models (classifiers and regressors). Currently these include four from _scikit-learn_ and three others:

* gradient boosting (GB)
* random forest (RF)
* extremely randomized trees (XT)
* linear SGD
* factorization machines (from polylearn)
* polynomial networks (from polylearn)
* a multilayer perceptron (from Keras)

Meta-classifier/regressor
-------------------------

Use `defs.meta`/`defs_regression.meta` to try many models in one Hyperband run. This is an automatic alternative to constructing search spaces with multiple models (like `defs.rf_xt`, or `defs.polylearn_fm_pn`) by hand.

Loading data
------------

Definitions files in `defs`/`defs_regression` import data from `load_data.py` and `load_data_regression.py`, respectively.

Edit these files, or a definitions file directly, to make your data available for tuning.

Regression defs use the _kin8nm_ dataset in `data/kin8nm`. There is no attached data for classification.

For the provided models data format follows _scikit-learn_ conventions, that is, there are _x_train_, _y_train_, _x_test_ and _y_test_ Numpy arrays.

Usage
-----

Run `main.py` (with your own data), or `main_regression.py`. The essence of it is

```python
from hyperband import Hyperband
from defs.gb import get_params, try_params

hb = Hyperband( get_params, try_params )
results = hb.run()
```

Here's a sample output from a run (three configurations tested) using `defs.xt`:

	3 | Tue Feb 28 15:39:54 2017 | best so far: 0.5777 (run 2)

	n_estimators: 5
	{'bootstrap': False,
	'class_weight': 'balanced',
	'criterion': 'entropy',
	'max_depth': 5,
	'max_features': 'sqrt',
	'min_samples_leaf': 5,
	'min_samples_split': 6}

	# training | log loss: 62.21%, AUC: 75.25%, accuracy: 67.20%
	# testing  | log loss: 62.64%, AUC: 74.81%, accuracy: 66.78%

	7 seconds.

	4 | Tue Feb 28 15:40:01 2017 | best so far: 0.5777 (run 2)

	n_estimators: 5
	{'bootstrap': False,
	'class_weight': None,
	'criterion': 'gini',
	'max_depth': 5,
	'max_features': 'sqrt',
	'min_samples_leaf': 1,
	'min_samples_split': 2}

	# training | log loss: 53.39%, AUC: 75.69%, accuracy: 72.37%
	# testing  | log loss: 53.96%, AUC: 75.29%, accuracy: 71.89%

	7 seconds.

	5 | Tue Feb 28 15:40:07 2017 | best so far: 0.5396 (run 4)

	n_estimators: 5
	{'bootstrap': True,
	'class_weight': None,
	'criterion': 'gini',
	'max_depth': 3,
	'max_features': None,
	'min_samples_leaf': 7,
	'min_samples_split': 8}

	# training | log loss: 50.20%, AUC: 77.04%, accuracy: 75.39%
	# testing  | log loss: 50.67%, AUC: 76.77%, accuracy: 75.12%

	8 seconds.
	
Early stopping
--------------

Some models may use early stopping (as the Keras MLP example does). If a configuration  stopped early, it doesn't make sense to run it with more iterations (duh). To indicate this, make `try_params()`

```python
return { 'loss': loss, 'early_stop': True }
```
	
This way, Hyperband will know not to select that configuration for any further runs.

Moar
----

See [http://fastml.com/tuning-hyperparams-fast-with-hyperband/](http://fastml.com/tuning-hyperparams-fast-with-hyperband/) for a detailed description.
