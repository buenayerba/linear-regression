# author: Yerbol Aussat

import numpy as np
import matplotlib.pyplot as plt

# loading training data
def load_training_samples():
	global trn_data
 	temp_trn_data = np.genfromtxt('housing_X_train.csv', delimiter=',')

 	#adding extra column of ones:
 	trn_data = np.c_[temp_trn_data, np.ones(temp_trn_data.shape[0])]
		
# loading training labels
def load_training_labels():
	global trn_labels
 	trn_labels = np.genfromtxt('housing_y_train.csv', delimiter=',')

# loading testing data
def load_testing_samples():
	global test_data
 	temp_test_data = np.genfromtxt('housing_X_test.csv', delimiter=',')

 	#adding extra column of ones:
 	test_data = np.c_[temp_test_data, np.ones(temp_test_data.shape[0])]
		
# loading testinglabels
def load_testing_labels():
	global test_labels
 	test_labels = np.genfromtxt('housing_y_test.csv', delimiter=',')


# ridge regression (to find vector w)
def ridge_regression(X, y, reg_const):
	A = np.dot(np.transpose(X), X) + reg_const * np.eye(X.shape[1])
	b = np.dot(np.transpose(X), y)
	w = np.linalg.solve(A, b)
	return w

# calculating square error
# calculating square error
def get_square_error(X, y, w):
	return np.mean( (np.dot(X, w) - y)**2 )
	
# trying different lambdas	
# input training and testing sets
def errors_for_different_lambdas(X_train, y_train, X_test, y_test, name):
	errors = []
	lambdas = np.linspace(0, 100, 11)
	
	percent_nonzeros = []
	for reg_const in lambdas:
		w = ridge_regression(X_train, y_train, reg_const)
		sqare_error = get_square_error(X_test, y_test, w)
		errors.append(sqare_error)
		percent_nonzeros.append(1.0 * np.count_nonzero(w != 0.0) / w.shape[0])		
# 	plt.plot(lambdas, errors)
# 	plt.ylabel('Mean Square Errors')
# 	plt.xlabel('Regularization Constant')
# 	plt.title(name)
# 	plt.grid(True)
#  	plt.show()
	return errors, percent_nonzeros

def find_valid_error(X_train, y_train):
	fold_size = np.linspace(0, 306, 11, dtype = int)
	errors = []
	percent_nonzeros = []
	lambdas = np.linspace(0, 100, 11)
	for reg_const in lambdas:
		
		# average squared error for regularization const
		square_errors_for_lambda = []
		percent_nonzeros_for_lambda = 0

		for i in range(10):
			# creating validation sets and reduced training sets that exclude validation set
			X_tr0 = np.delete(X_train, range(fold_size[i], fold_size[i+1]), 0) 
			y_tr0 = np.delete(y_train, range(fold_size[i], fold_size[i+1]), 0) 
			X_valid = X_train[fold_size[i]:fold_size[i+1], :]
			y_valid = y_train[fold_size[i]:fold_size[i+1]]
	
			# determining mean square error for each validation set
			w = ridge_regression(X_tr0, y_tr0, reg_const)
			sqare_error = get_square_error(X_valid, y_valid, w)
			square_errors_for_lambda.append(sqare_error)
			percent_nonzeros_for_lambda += 1.0 * np.count_nonzero(w != 0.0) / w.shape[0]
			
		percent_nonzeros.append(percent_nonzeros_for_lambda / 10)
		square_error_ave = np.mean(square_errors_for_lambda)
		errors.append(square_error_ave)
	
# 	plt.plot(lambdas, errors)
# 	plt.ylabel('Mean Square Errors')
# 	plt.xlabel('Regularization Constant')
# 	plt.title('MSE on validation set')
# 	plt.grid(True)
#  	plt.show()
	return errors, percent_nonzeros
	
def main():
	load_training_samples()
	load_training_labels()
	load_testing_samples()
	load_testing_labels()
	
 	trn_data_altered = np.concatenate((trn_data, np.random.standard_normal([trn_data.shape[0],1000])), axis = 1)
 	test_data_altered = np.concatenate((test_data, np.random.standard_normal([test_data.shape[0],1000])), axis = 1)
		
	# for training set:
	errors1, percent_nonzeros1 = errors_for_different_lambdas(trn_data_altered, trn_labels, trn_data_altered, trn_labels, 'MSE on training set')
 	print "\ntraining set"
 	for i in range(len(errors1)):
 		print "%.2f" %errors1[i]
 	print "percentage of nonzeros"
 	for i in range(len(percent_nonzeros1)):
 		print percent_nonzeros1[i] * 100
 	
 	# for test set:
 	errors2, percent_nonzeros2 = errors_for_different_lambdas(trn_data_altered, trn_labels, test_data_altered, test_labels, 'MSE on test set')
 	print "\ntest set"
 	for i in range(len(errors2)):
 		print "%.2f" %errors2[i] 	
 	print "percentage of nonzeros"
 	for i in range(len(percent_nonzeros2)):
 		print percent_nonzeros2[i] * 100

	# for validation set:

	errors3, percent_nonzeros3 = find_valid_error(trn_data_altered, trn_labels)
 	print "\nvalidation set"
 	for i in range(len(errors3)):
 		print "%.2f" %errors3[i] 	
 	print "percentage of nonzeros"
 	for i in range(len(percent_nonzeros3)):
 		print percent_nonzeros3[i] * 100
	
if __name__ == "__main__":
    main()