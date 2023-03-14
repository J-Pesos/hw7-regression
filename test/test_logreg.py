"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression import logreg, utils
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# (you will probably need to import more things here)

def test_prediction():
	'''Test prediction
	Tests to make sure implemented model provides the same number of predictions and that
	prediction values make sense.
	'''
	# Load sample dataset.
	X_train, X_val, y_train, y_val = utils.loadDataset(split_percent=0.8)

	# Scale the data, since values vary across feature. Note that we
    # fit on the training data and use the same scaler for X_val.
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform (X_val)

	# For testing purposes, once you've added your code.
    # CAUTION: hyperparameters have not been optimized.
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)

	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	preds = log_model.make_prediction(X_val)[0:10]

	# Check to make sure model provides expected number of predictions.
	assert np.shape(preds) == (10,), 'Model is not providing correct number of predictions.'

	# Check that precitions are all between 0 and 1.
	assert len(preds[(preds <= 1) & (preds >= 0)]) == len(preds), 'Predictions do not fall between values of 0 and 1.'
	

def test_loss_function():
	'''Test loss function
	Compares loss of implemented method against sklearn method.
	'''
	# Load sample dataset.
	X_train, X_val, y_train, y_val = utils.loadDataset(split_percent=0.8)

	# Scale the data, since values vary across feature. Note that we
    # fit on the training data and use the same scaler for X_val.
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform (X_val)

	# For testing purposes, once you've added your code.
    # CAUTION: hyperparameters have not been optimized.
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)

	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

	# Compare loss in my method to sklearn's method.
	my_loss = log_model.loss_function(y_val, log_model.make_prediction(X_val))
	sklearn_loss = log_loss(y_val, log_model.make_prediction(X_val))
	assert np.isclose(my_loss, sklearn_loss), 'Your method does not match sklearn method loss.'

def test_gradient():
	'''Test gradient
	Ensure that the gradient matches the number of features.
	'''
	# Load sample dataset.
	X_train, X_val, y_train, y_val = utils.loadDataset(split_percent=0.8)

	# Scale the data, since values vary across feature. Note that we
    # fit on the training data and use the same scaler for X_val.
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform (X_val)

	# For testing purposes, once you've added your code.
    # CAUTION: hyperparameters have not been optimized.
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)

	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	gradient = log_model.calculate_gradient(y_train, X_train)

	# Check that the shape of the gradient is equal to the number of features.
	print(gradient.shape[0])
	assert gradient.shape[0] == log_model.num_feats

def test_training():
	'''Test training
	Check to make sure that weights are being updated during the course of your model training.
	'''
	# Load sample dataset.
	X_train, X_val, y_train, y_val = utils.loadDataset(split_percent=0.8)

	# Scale the data, since values vary across feature. Note that we
    # fit on the training data and use the same scaler for X_val.
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform (X_val)

	# For testing purposes, once you've added your code.
    # CAUTION: hyperparameters have not been optimized.
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
	log_model.train_model(X_train, y_train, X_val, y_val)

	assert (len(log_model.loss_hist_train) == len(log_model.loss_hist_val)), 'Weights are not being updated during model training.'