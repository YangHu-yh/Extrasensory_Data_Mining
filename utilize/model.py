import numpy as np
from numpy import linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler, Adam

from sklearn.base import ClassifierMixin

from utilize.test import evaluate_model
alpha = 0.0001
class MLP(nn.Module):
	# MLP model with two hidden layers realized by pytorch 

	def __init__(self, n_hidden_layers, n_features, n_labels, dropout1, dropout2, batchnorm1, batchnorm2, alpha):
		# Initialization: take n_hidden_layers, n_feaures and n_labels as input 
		super(MLP, self).__init__()
		self.n_features = n_features
		self.n_labels = n_labels
		self.n_hidden_layers = n_hidden_layers
		self.batchnorm1 = batchnorm1
		self.batchnorm2 = batchnorm2
		# fc1, fc2 two hidden layers, fc3 output layer
		self.fc0_bn = nn.BatchNorm1d(num_features=n_features)
		self.fc1 = nn.Linear(n_features, n_hidden_layers[0])
		self.fc1_bn = nn.BatchNorm1d(num_features=n_hidden_layers[0])
		self.fc1_do = nn.Dropout(p = dropout1)
		self.fc2 = nn.Linear(n_hidden_layers[0], n_hidden_layers[1])
		self.fc2_bn = nn.BatchNorm1d(num_features=n_hidden_layers[1])
		self.fc2_do = nn.Dropout(p = dropout2)
		self.fc3 = nn.Linear(n_hidden_layers[1], n_labels)

	def forward(self, x):
		if self.batchnorm1=='On':
			x = self.fc0_bn(x)
		x = self.fc1(x)
		x = F.prelu(x, torch.tensor(0.1))
		if self.batchnorm1=='On':
			x = self.fc1_bn(x)
		x = self.fc1_do(x)
		x = self.fc2(x)
		x = F.prelu(x, torch.tensor(0.1))
		if self.batchnorm2=='On':
			x = self.fc2_bn(x)
		x = self.fc2_do(x)
		x = self.fc3(x)

		return x

	def logist(self, x):
		# from outputs compute the probability
		x = torch.sigmoid(x)

		return x
		# Add dropout layers and batch nomarlization layers

class MLP_model(ClassifierMixin):
	# MLP model compatible with sklearn etimator API
	def __init__(self, n_hidden_layers, target_labels, epoches = 20, learning_rate = 0.00001, batch_size = 300, dropout1 =0.5, dropout2 = 0.5, batchnorm1= 'On', batchnorm2 ='On', alpha = 0.0001):
		'''
		Initialze define the parameters here. 
		Keyword Arguments:
			n_hidden_layers: [integer list] -- take the first two numbers as the size of two hidden units
			target_labels: [string list] -- target labels we used for our multi-label classfication problems
			epoches: [int] -- number of epoches we used to do a single fit (training) 
			learning_rate: [float] -- learning rate of the Adam optimizer 
			score: [string] -- name of the scorer, default is 'BA', balanced accuracy 
		'''

		self.target_labels = target_labels
		self.n_hidden_layers = n_hidden_layers
		self.epoches = epoches
		self.learning_rate = learning_rate 
		#self.score = score 
		self.batch_size = batch_size
		self.dropout1 = dropout1
		self.dropout2 = dropout2
		self.batchnorm1 = batchnorm1
		self.batchnorm2 = batchnorm2
		self.alpha = alpha

	def predict(self, X):
		# Predict labels for given instances 

		if type(X) is not torch.Tensor:
			X = torch.tensor(X).float()
		y_out = self.MLP(X)
		y_pred = self.MLP.logist(y_out) > 0.5

		return y_pred

	def score(self, X, y, M = None):
		# score the model given instances and true labels
		W = abs(1-M)
		accuracy, sensitivity, specificity, BA = evaluate_model(self, X, y, W)

		
		return BA
		

	def fit(self, X_train, y_train, X_test = None, y_test = None, M_train = None, M_test = None, report = False):
		'''
		fit the model
		Keyword Arguments:
			X_train, y_train, M_train: feature matrix, label matrix and missing label matrix for the training set
			X_test, y_test, M_test: feature matrix, label matrix and missing label matrix for the test set. 
									None means it will not report the test score during the training, 
									M_test == None means that the missing labels will not be accounted durin the test
			report: whether to report training loss and test loss during the traininig process
			M_train: default None means that the weighting matrix will not count the effect of missing labels
		'''
		# Initialize the model by parameters 
		self.MLP = MLP(self.n_hidden_layers, X_train.shape[1], y_train.shape[1], dropout1 = self.dropout1,dropout2 = self.dropout2, batchnorm1 = self.batchnorm1, batchnorm2 = self.batchnorm2, alpha = self.alpha)#[1] number of feature, number of label

		# build the instance weighting matrix
		if M_train is not None:
			# Count both the effect of imbalanced classes and missing labels
			W_train = (y_train/np.sum(y_train, axis = 0)/2 + abs(y_train -1)/(np.sum(abs(y_train -1), axis = 0))/2)*y_train.shape[0]*abs(M_train-1)
		else:
			W_train = (y_train/np.sum(y_train, axis = 0)/2 + abs(y_train -1)/(np.sum(abs(y_train -1), axis = 0))/2)*y_train.shape[0]

		# In order to train using pytorch, convert the data to torch tensor
		X_train = torch.tensor(X_train).float()
		y_train = torch.tensor(y_train).float()
		W_train = torch.tensor(W_train).float()

		if M_test is not None:
			W_test = torch.tensor(np.abs(M_test-1)).float()
			X_test = torch.tensor(X_test).float()
			y_test = torch.tensor(y_test).float()


		batch_size = self.batch_size

		# define the optimizer and loss function 
		optimizer = Adam(self.MLP.parameters(), lr=self.learning_rate)
		BCE = F.binary_cross_entropy

		# train the model for epoches times
		for i in range(self.epoches):
			# shuffle the whole training set every time
			permutation = np.random.permutation(X_train.shape[0])
			X_train = X_train[permutation, :]
			y_train = y_train[permutation, :]
			W_train = W_train[permutation, :]
			# For each mini-batch do a single forward propagation and backward propagation
			for batch_idx in range(int(X_train.shape[0]/batch_size)):

				X_batch = X_train[batch_idx*batch_size:(batch_idx+1)*batch_size, :]
				y_batch = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size, :]
				W_batch = W_train[batch_idx*batch_size:(batch_idx+1)*batch_size, :]

				y_out = self.MLP(X_batch)
				y_out = self.MLP.logist(y_out)
				loss = BCE(y_out, y_batch, weight = W_batch)+alpha*np.linalg.norm(W_batch, 'fro')

				loss.backward()
				optimizer.step()
				#scheduler.step()

				# Report the training error every 100 mini-batch
				if (batch_idx+1) % 100 == 0 and report:
					print('Train Epoch: %d [%d/%d (%d%%)]\tLoss: %.6f' %(i, batch_idx * batch_size, X_train.shape[0], 100. * batch_idx * batch_size/ X_train.shape[0], loss.item()))
			# report the test score when an epoch is finished
			if X_test is not None and report:
				print('Test epoch %d:' %(i))
				evaluate_model(self, X_test, y_test, W_test)

	# Following are API to meet the sklearn custom estimator, don't have to understand 
	def setattr(self, parameter, value): 

		if parameter == 'target_labels':
			self.target_labels = value
		elif parameter == 'n_hidden_layers':
			self.n_hidden_layers = value
		elif parameter == 'epoches':
			self.epoches = value
		elif parameter == 'learning_rate':
			self.learning_rate = value
		elif parameter == 'score':
			self.score = value

	def get_params(self, deep=True):

		return {"target_labels": self.target_labels, "n_hidden_layers": self.n_hidden_layers, "epoches": self.epoches, "learning_rate": self.learning_rate, "score": self.score}

	def set_params(self, **parameters):

		for parameter, value in parameters.items():
			setattr(self, parameter, value)

		return self 

if __name__ == 'main': 

	# Didn't include the code for loading the data
	# Just for your reference about how to use the model

	# Shows how we use it here 
	mlp = MLP_model([64, 64], target_label)
	mlp.fit(X_train, y_train, X_test, y_test, M_train, M_test, epoches = 20, learning_rate = 0.00001, report = True)
