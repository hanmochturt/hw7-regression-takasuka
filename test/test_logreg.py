"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

import pytest
import numpy as np
import sklearn
import sys
import pathlib

PARENT_PARENT_FOLDER = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PARENT_PARENT_FOLDER))
from regression import (logreg, utils)

X = np.expand_dims(np.arange(0.0, 4.0, 0.1), axis=1)
WEIGHT = 100
coefficient = -101
Y_TRUE = np.zeros(X.shape)
Y_TRUE[X > -coefficient/WEIGHT] = 1
Y_TRUE = np.squeeze(Y_TRUE)


def test_prediction():
	"""Unit test that the prediction function correctly maps logistic weights to the X values"""
	log_model = logreg.LogisticRegressor(num_feats=X.shape[1], learning_rate=0.00001, tol=0.01,
										 max_iter=10,
										 batch_size=10)
	log_model.W = [WEIGHT, coefficient]
	y_pred = log_model.make_prediction(X)
	y_expected = 1/(1+np.exp(-WEIGHT*X-coefficient)).T
	assert np.allclose(y_expected, y_pred)


def test_loss_function():
	"""Unit test that the loss function correctly calculates average loss very close to zero when
	the true weights are applied to the prediction function"""
	log_model = logreg.LogisticRegressor(num_feats=X.shape[1], learning_rate=0.00001, tol=0.01,
										 max_iter=10, batch_size=10)
	log_model.W = [WEIGHT, coefficient]
	y_pred = log_model.make_prediction(X)
	assert log_model.loss_function(Y_TRUE, y_pred) < 0.01


def test_gradient():
	"""Unit test that the gradient correctly calculates to bve above zero when weights above
	their true values are applied"""
	log_model = logreg.LogisticRegressor(num_feats=X.shape[1], learning_rate=0.00001, tol=0.01,
										 max_iter=10, batch_size=10)
	log_model.W = [WEIGHT+1, coefficient+1]
	grad = log_model.calculate_gradient(Y_TRUE, X)
	assert all(i > 0 for i in grad)


def test_training():
	"""Unit test that the training algorithm changes the random weights in the correct direction
	towards the true weights"""
	log_model = logreg.LogisticRegressor(num_feats=X.shape[1], learning_rate=0.01, tol=0.001,
										 max_iter=1000, batch_size=100)
	random_weights = log_model.W
	X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y_TRUE)
	log_model.train_model(X_train, y_train, X_test, y_test)
	true_weights = [WEIGHT, coefficient]
	for index, weight in enumerate(log_model.W):
		lower_bound = min(true_weights[index], random_weights[index])
		upper_bound = max(true_weights[index], random_weights[index])
		assert lower_bound <= weight <= upper_bound
