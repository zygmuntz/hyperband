hyperband
=========

Code for tuning hyperparams with Hyperband, adapted from [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://people.eecs.berkeley.edu/~kjamieson/hyperband.html). 

	common_defs.py - imports and definitions shared by defs files
	defs/ - functions and search space definitions for various classifiers
	hyperband.py - from hyperband import Hyperband
	load_data.py - data is a dict with x_train, y_train, x_test, y_test
	main.py - a complete example
	main_simple.py - a simple example	

The goal is to provide a fully functional implementation of Hyperband, as well as a number of ready to use functions for a number of classifiers. Currently these include a few scikit-learn classifiers:

* gradient boosting (GB)
* random forest (RF)
* extremely randomized trees (XT)
* linear SGD classifier

Plus one Keras-based multilayer perceptron, and two [polylearn](https://github.com/scikit-learn-contrib/polylearn/) classifiers. 

Meta-classifier
---------------

Use `defs.meta` to try many classifiers in one Hyperband run. This is an automatic alternative to constructing search spaces with multiple classifiers, like `defs.rf_xt`, or `defs.polylearn_fm_pn`, by hand.

Usage
-----

Edit `load_data.py`, or a definitions file directly, to provide your data. Then run `main.py`. The essence of it is

```python
from hyperband import Hyperband
from defs.gb import get_params, try_params

hb = Hyperband( get_params, try_params )
results = hb.run()
```

Sample output from a run (three configurations tested) using `defs.xt`:

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

See [http://fastml.com/tuning-hyperparams-fast-with-hyperband/](http://fastml.com/tuning-hyperparams-fast-with-hyperband/) for a detailed description.
