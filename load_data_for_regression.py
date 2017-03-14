import numpy as np

train_file = 'data/kin8nm/train.csv'
valid_file = 'data/kin8nm/validation.csv'
test_file = 'data/kin8nm/test.csv'

print "loading data..."

train = np.loadtxt( open( train_file ), delimiter = "," )
valid = np.loadtxt( open( valid_file ), delimiter = "," )
#test = np.loadtxt( open( test_file ), delimiter = "," )

y_train = train[:,-1]
y_test = valid[:,-1]
#y_test = test[:,-1]

x_train = train[:,0:-1]
x_test = valid[:,0:-1]
#x_test = test[:,0:-1]

data = { 'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test }
