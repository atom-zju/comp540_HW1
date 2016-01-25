# -*- coding: utf-8 -*-
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
from sklearn import cross_validation
import plot_utils
from reg_linear_regressor_multi import RegularizedLinearReg_SquaredLoss
from utils import feature_normalize
import sklearn
from sklearn.preprocessing import PolynomialFeatures



# load data
bdata = load_boston()
df = pd.DataFrame(data=bdata.data, columns=bdata.feature_names)
X = df.values
y = bdata.target


# split the data into train set, validation set and test set
X_train_val, X_test, y_train_val, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = cross_validation.train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=4)


# prepare linear data
X_train_norm, mu, sigma = utils.feature_normalize(X_train)
XX_train_norm = np.vstack([np.ones((X_train_norm.shape[0],)),X_train_norm.T]).T

X_test_norm = (X_test - mu) / sigma
X_val_norm = (X_val - mu) / sigma
XX_test_norm = np.vstack([np.ones((X_test_norm.shape[0],)),X_test_norm.T]).T
XX_val_norm = np.vstack([np.ones((X_val_norm.shape[0],)),X_val_norm.T]).T


# determine the test error with lambda =0
reg = 0
reglinear_reg1 = RegularizedLinearReg_SquaredLoss()
theta_opt1 = reglinear_reg1.train(XX_train_norm,y_train,reg=reg,num_iters=10000)
print 'Theta at lambda = 0 is ', theta_opt1
print 'Test err of the best linear model with lambda=0 is: '+str(reglinear_reg1.loss(theta_opt1, XX_test_norm, y_test, 0))

# plot the validation curve to determine the best lambda
reg_vec, error_train, error_val = utils.validation_curve(XX_train_norm,y_train,XX_val_norm,y_val)
plot_utils.plot_lambda_selection(reg_vec,error_train,error_val)
plt.savefig('boston_linear_val_cuv.pdf')

# determine the lowest test error with linear model
reg = reg_vec[error_val.tolist().index(min(error_val))]
print 'the best lambda is:',reg
reglinear_reg1 = RegularizedLinearReg_SquaredLoss()
theta_opt1 = reglinear_reg1.train(XX_train_norm,y_train,reg=reg,num_iters=10000)
print 'Test err of the best linear model with lambda='+str(reg)+' is: '+str(reglinear_reg1.loss(theta_opt1, XX_test_norm, y_test, 0))

# prepare quadratic data
p = 2
poly = sklearn.preprocessing.PolynomialFeatures(degree=p,include_bias=False)

X_quad_train = poly.fit_transform(X_train)
X_quad_val = poly.fit_transform(X_val)
X_quad_test = poly.fit_transform(X_test)

X_quad_train_norm, mu2, sigma2 = utils.feature_normalize(X_quad_train)


XX_quad_train_norm = np.vstack([np.ones((X_quad_train_norm.shape[0],)),X_quad_train_norm.T]).T
X_quad_test_norm = (X_quad_test - mu2) / sigma2
X_quad_val_norm = (X_quad_val - mu2) / sigma2
XX_quad_test_norm = np.vstack([np.ones((X_quad_test_norm.shape[0],)),X_quad_test_norm.T]).T
XX_quad_val_norm = np.vstack([np.ones((X_quad_val_norm.shape[0],)),X_quad_val_norm.T]).T


# plot the validation curve to determine the best lambda
reg_vec, error_train, error_val = utils.validation_curve(XX_quad_train_norm,y_train,XX_quad_val_norm,y_val)
plot_utils.plot_lambda_selection(reg_vec,error_train,error_val)
plt.savefig('boston_quad_val_cuv.pdf')

# determine the lowest test error with quadratic model
reg = reg_vec[error_val.tolist().index(min(error_val))]
print 'the best lambda is:',reg
reglinear_reg2 = RegularizedLinearReg_SquaredLoss()
theta_opt2 = reglinear_reg2.train(XX_quad_train_norm,y_train,reg=reg,num_iters=10000)
print 'Test err of the best quadratic model with lambda='+str(reg)+' is: '+str(reglinear_reg2.loss(theta_opt2, XX_quad_test_norm, y_test, 0))


# prepare cubic data
p = 3
poly = sklearn.preprocessing.PolynomialFeatures(degree=p,include_bias=False)


X_cubic_train = poly.fit_transform(X_train)
X_cubic_val = poly.fit_transform(X_val)
X_cubic_test = poly.fit_transform(X_test)

X_cubic_train_norm, mu2, sigma2 = utils.feature_normalize(X_cubic_train)

XX_cubic_train_norm = np.vstack([np.ones((X_cubic_train_norm.shape[0],)),X_cubic_train_norm.T]).T
X_cubic_test_norm = (X_cubic_test - mu2) / sigma2
X_cubic_val_norm = (X_cubic_val - mu2) / sigma2
XX_cubic_test_norm = np.vstack([np.ones((X_cubic_test_norm.shape[0],)),X_cubic_test_norm.T]).T
XX_cubic_val_norm = np.vstack([np.ones((X_cubic_val_norm.shape[0],)),X_cubic_val_norm.T]).T

# plot the validation curve to determine the best lambda
reg_vec, error_train, error_val = utils.validation_curve(XX_cubic_train_norm,y_train,XX_cubic_val_norm,y_val)
plot_utils.plot_lambda_selection(reg_vec,error_train,error_val)
plt.savefig('boston_cubic_val_cuv.pdf')

# determine the lowest test error with cubic model
reg = reg_vec[error_val.tolist().index(min(error_val))]
print 'the best lambda is:',reg
reglinear_reg2 = RegularizedLinearReg_SquaredLoss()
theta_opt2 = reglinear_reg2.train(XX_cubic_train_norm,y_train,reg=reg,num_iters=10000)
print 'Test err of the best cubic model with lambda='+str(reg)+' is: '+str(reglinear_reg2.loss(theta_opt2, XX_cubic_test_norm, y_test, 0))
