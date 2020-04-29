from utilize.transform import select_features_by_name
from utilize.test import evaluate_model

import keras
from keras.models import Model
from keras.layers import Dense,Input
import keras.backend as K
import numpy as np
from keras.callbacks import EarlyStopping

class feature_selector():

	def __init__(self, feature_names):

		self.feature_names = feature_names
		self.selected_features = []

	def forward_sequential_selection(self, n_features, X_train, y_train, X_test, y_test, model, 
										evaluation = 'BA', report = 'True'):
		'''
		Take splitted dataset and a model, use forward sequential selection to select a 
		subset of features that gives the highest score on given test set. 

		Keyword Arguments:
			X_train, y_train, X_test, y_test: [narray] -- splitted dataset
			model: [sklearn model] -- model used to fit and predict the dataset
			evaluation: [str] -- specify the type of evaluation (score)
			report: [Boolen] -- whether to report the progress

		'''

		self.n_labels = y_train.shape[-1]
		self.n_features = n_features 
		self.evaluation = evaluation

		selected_features = self.selected_features
		remaining_features = self.feature_names.copy()
		for feature in selected_features:
			remaining_features.remove(feature)
		score = 0

		for i in range(n_features):

			# Try all the features that has not been selected
			for j, feature in enumerate(remaining_features):

	        	# Select features
				temp_features = selected_features + [feature]
				feature_selector = select_features_by_name(temp_features, self.feature_names)
				temp_X_train = feature_selector.fit_transform(X_train)
				temp_X_test = feature_selector.transform(X_test)

				# Fit the model
				model.fit(temp_X_train, y_train)

				# predict and calcuate score
				if evaluation == 'model_default':
					temp_score = model.score(temp_X_test, y_test)
				elif evaluation == 'BA':
					_, _, _, temp_score = evaluate_model(model, temp_X_test, y_test, report = False)
				else:
					raise NameError('Evaluation does not exist!')
	            
				# If the score increases, update 
				if temp_score > score:
					score = temp_score
					added_feature = feature
					
				print('Try feature: %s\t[%d/%d]\t score: %f' %(feature, j, len(remaining_features), temp_score))
	                
	        # Update current selected features and remaining feature to be selected
			selected_features = selected_features + [added_feature]
			remaining_features.remove(added_feature)

			if report:
				print('\nAdd %s\t%d features selected\tscore %f' %(added_feature, len(selected_features), score))
	    
		self.selected_features = selected_features

		return selected_features, score

	def backward_sequential_selection(self, n_features, X_train, y_train, X_test, y_test, model, 
										evaluation = 'BA', report = 'True', feature_increment = 1):
		'''
		Take splitted dataset and a model, use forward sequential selection to select a 
		subset of features that gives the highest score on given test set. 

		Keyword Arguments:
			X_train, y_train, X_test, y_test: [narray] -- splitted dataset
			model: [sklearn model] -- model used to fit and predict the dataset
			evaluation: [str] -- specify the type of evaluation (score)
			report: [Boolen] -- whether to report the progress

		'''

		self.n_labels = y_train.shape[-1]
		self.n_features = n_features 
		self.evaluation = evaluation

		selected_features = self.feature_names.copy()

		for i in range(int((len(self.feature_names) - n_features)/feature_increment)):
			dropped_features_dict = {}

			# Try all the features that has not been selected
			for j, feature in enumerate(selected_features):

		        # Select features
				temp_features = selected_features.copy()
				temp_features.remove(feature)
				feature_selector = select_features_by_name(temp_features, self.feature_names)
				temp_X_train = feature_selector.fit_transform(X_train)
				temp_X_test = feature_selector.transform(X_test)

				# Fit the model
				model.fit(temp_X_train, y_train)

				# predict and calcuate score
				if evaluation == 'model_default':
					temp_score = model.score(temp_X_test, y_test)
				elif evaluation == 'BA':
					_, _, _, temp_score = evaluate_model(model, temp_X_test, y_test, report = False)
				else:
					raise NameError('Evaluation does not exist!')
		            
				# If the score increases, update 
				# if temp_score > score:
				#	score = temp_score
				#	dropped_feature = feature

				dropped_features_dict.update([(feature, temp_score)])
					

				print('Try feature: %s\t[%d/%d]\t score: %f' %(feature, j, len(selected_features), temp_score))
		                
		    # Update current selected features and remaining feature to be selected
			dropped_features = sorted(dropped_feature, key=lambda x: int(dropped_features_dict[x]), reverse = True)[:feature_increment]
			for feature in dropped_features:
				selected_features.remove(feature)
				if report:
					print('\nAdd %s\t%d features selected\tscore %f' %(feature, len(selected_features), dropped_features_dict[feature]))
		    
		self.selected_features = selected_features

		return selected_features

class RRFS():
    
    def __init__(self, dim, l2 = .001, hidden=20, loss = 'mse'):
        
        self.l2 = l2
        self.early_stopping = EarlyStopping(patience=3)
        x1 = Input((dim,))
        x2 = Dense(hidden, activation = 'relu', kernel_regularizer = keras.regularizers.l2(l2))(x1)
        x3 = Dense(dim, kernel_regularizer = keras.regularizers.l2(l2))(x2)
        self.autoencoder = Model(x1, x3)
        self.encoder = Model(x1, x2)
        self.autoencoder.compile(optimizer = 'Adam', loss = loss)
        
    def train_autoencoder(self, X, X_val, batch_size=300, epochs = 100, evaluate = False):
        self.autoencoder.fit(X, X, epochs = epochs, batch_size = batch_size, 
                             callbacks = [self.early_stopping], 
                             validation_data = (X_val, X_val), verbose = 1)
        if evaluate:
            loss = self.autoencoder.evaluate(X, X)
            return loss
        
        self.encoder.set_weights(self.autoencoder.layers[1].get_weights())
    
    def feature_scores(self): 
        w = self.autoencoder.layers[1].get_weights()[0]
        w = np.sum(np.square(w),1)
        
        return w
    
    def encode(self, X): 
        
        return self.encoder.predict(X)
    
    def feature_index(self, k_features = 175):
        
        scores = self.feature_scores
        idx = np.argsort(scores)[::-1][:175]
        
        return idx

if __name__ == 'main': 

	None