from __future__ import print_function, division, absolute_import
#%matplotlib inline
from matplotlib import pyplot as plt
import numpy as np
import math
import os
import argparse
import time
import sklearn
import scipy.io as sio
from scipy.linalg import sqrtm
from plot import plot_images, plot_embedding, plot_embedding_annotation, plot_confusion_matrix
from dataset import read_mnist, read_mnist_twoview
from utils import resize
from svm import linear_svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

"""
CCA
"""
"""TEST CASES
#Create some test matrices
test1 = np.array([[1,1,0,3,1, 2,2,3], [1,2, 1,2,2,1,3, 2], [3,4, 2,3,3,3,3, 4],
                 [ 4,4,5,4,5, 3,3,5], [4,3, 5,5,6,4,6, 5]])
test2 = np.array([[ 1,2,0,0,1, 2], [ 0,2,3,1,3,2], [ 2,2,3,3,3,4],
                 [ 5,4,3,4,5, 3], [ 4,5,4,6,5,5]])
"""

np.random.seed(1)

datapath="data/noisy_two_view_distribute.mat"
print("Data path is: %s" % datapath)

trainData,tuneData,testData1,testData2=read_mnist_twoview(datapath)
train_x_sample1 = trainData.images
train_x_sample2 = trainData.images2
train_x_image1 = np.reshape(train_x_sample1, [train_x_sample1.shape[0],28,28]).transpose(0, 2, 1)
train_x_image2 = np.reshape(train_x_sample2, [train_x_sample2.shape[0],28,28]).transpose(0, 2, 1)
train_y_sample = np.reshape(trainData.labels, [train_x_sample1.shape[0]])
tune_x_sample1 = tuneData.images
tune_x_sample2 = tuneData.images2
tune_x_image1 = np.reshape(tune_x_sample1, [tune_x_sample1.shape[0],28,28]).transpose(0, 2, 1)
tune_x_image2 = np.reshape(tune_x_sample2, [tune_x_sample2.shape[0],28,28]).transpose(0, 2, 1)
tune_y_sample = np.reshape(tuneData.labels, [tune_x_sample1.shape[0]])

"""
Visualize a few input digits
"""
ax = plot_images(train_x_image1[::1500], 2, 2, 28, 28)
plt.show()

"""
Visualize more input digits, using smaller scale
"""
train_x_rescale = resize(train_x_image1,10,10)
ax = plot_images(train_x_rescale[::100], 10, 10, 10, 10)
plt.show()

"""
3. Reduce dimension with CCA
"""
class CCA(object):
    def __init__(self, n_components, r1, r2):
        """
        Set up CCA
        ------
        :in:        
        n_components: number of components to keep
        r1, r2: regularization coefficient on view 1 and view 2
        """
        self.n_components, self.r1, self.r2 = n_components, r1, r2

    def fit(self, X, Y):
        """
        TODO: fit the model with training data
        --------
        :in: 
        X, 2d-array of shape (n_samples, view1_features): data matrix of view 1
        Y, 2d-array of shape (n_samples, view2_features): data matrix of view 2
        """
        n = X.shape[0]
        nx_features, ny_features = X.shape[1], Y.shape[1]
        total_features = nx_features+ny_features
        self.mX, self.mY = np.zeros(nx_features), np.zeros(ny_features)
        self.mX = X.mean(axis=0)
        self.mY = Y.mean(axis=0)
        scaled_X = X - self.mX
        scaled_Y = Y - self.mY
        
        self.Wx, self.Wy = np.zeros((nx_features, self.n_components)), np.zeros((ny_features, self.n_components))

        #Set up the variables required for EVD
        XY = np.concatenate((scaled_X, scaled_Y), axis=1)        
        cov_X = np.cov(scaled_X.T)
        cov_Y = np.cov(scaled_Y.T)
        x_rI = self.r1 * np.identity(nx_features)
        y_rI = self.r2 * np.identity(ny_features)
        
        inverse_cov_X = np.linalg.inv(cov_X + x_rI)
        inverse_cov_Y = np.linalg.inv(cov_Y + y_rI)
        cov_full = np.cov(XY.T)
        cov_XY = cov_full[0:nx_features, nx_features:total_features]
        cov_YX = cov_XY.T
        
        #Wx        
        final_mat_x= inverse_cov_X @ cov_XY @ inverse_cov_Y @ cov_YX
        eig_vals_x, eig_vecs_x = np.linalg.eig(final_mat_x)
        eig_pairs_x = [(np.abs(eig_vals_x[i]), eig_vecs_x[:,i]) for i in range(len(eig_vals_x))]
        
        for j in range(self.n_components):
            ranked_eig_val_x, ranked_eig_vecs_x = eig_pairs_x[j]
            self.Wx[:,j] = ranked_eig_vecs_x

        #Wxy       
        final_mat_y = inverse_cov_Y @ cov_YX @ inverse_cov_X @ cov_XY
        eig_vals_y, eig_vecs_y = np.linalg.eig(final_mat_y)
        eig_pairs_y = [(np.abs(eig_vals_y[i]), eig_vecs_y[:,i]) for i in range(len(eig_vals_y))]
        
        for k in range(self.n_components):
            ranked_eig_val_y, ranked_eig_vecs_y = eig_pairs_y[k]
            self.Wy[:,k] = ranked_eig_vecs_y
        
        return

    def transform(self, X, view=1):
        """
        TODO: apply CCA on data X of a given view
        --------
        :in: X, 2d-array of shape (n_samples, n_features): data matrix
        view: view index (1 or 2)
        :out: Z, 2d-array of shape (n_samples, n_components): feature matrix
        """
        if view == 1:
            scaled_X = X - self.mX
            self.Z = scaled_X @ self.Wx
        else:
            scaled_Y = X - self.mY
            self.Z = scaled_Y @ self.Wy
        return self.Z
        
        
    def reconstruct(self, Z, view=1):
        """
        TODO: transform feature Z back to its original space
        --------
        :in: Z, 2d-array of shape (n_samples, n_components): feature matrix
        :out: X_hat, 2d-array of shape (n_samples, n_features): reconstructed data matrix
        """
        if view == 1:
            wz = Z @ self.Wx.T
            self.hat = self.mX + wz
        else:
            wz = Z @ self.Wy.T
            self.hat = self.mY + wz
        return self.hat

        
n_components, r1, r2 = 20, 1, 1
cca = CCA(n_components, r1, r2)

# train
cca.fit(train_x_sample1, train_x_sample2)

# transform
z_cca_train = cca.transform(train_x_sample1)
z_cca_tune = cca.transform(tune_x_sample1)

# reconstruct
tune_x_image_hat = cca.reconstruct(z_cca_tune).reshape((-1, 28, 28)).transpose((0, 2, 1))
ax = plot_images(tune_x_image_hat[::200], 4, 4, 28, 28)
plt.show()

# Visualization without annotation
vis_sample_rate = 10
plot_embedding(z_cca_tune[::vis_sample_rate, :2], tune_y_sample[::vis_sample_rate])
plt.show()

# Visualization with annotation
tune_x_rescale = resize(tune_x_image_hat,10,10)
plot_embedding_annotation(z_cca_tune[::vis_sample_rate, :2],tune_x_rescale[::vis_sample_rate],0.001)
plt.show()


"""
3-2. Classify with SVM
"""
"""
Train on raw view1 data 
"""
train1_x, tune1_x,train1_y,tune1_y = train_test_split(train_x_sample1,train_y_sample,
                                                                     train_size=.75,random_state=1)

best_error_tune,raw_pred = linear_svm(train1_x, train1_y, 
                                           tune1_x, tune1_y, tune_x_sample1,
                                            cc=[0.05, 0.1, 0.15])

class_names = np.array([0,1,2,3,4,5,6,7,8,9])
cnf_matrix = confusion_matrix(tune_y_sample, raw_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True)
plt.show()
#Find accuracy when SVM applied onto development set
error_rate = 1 - (np.matrix.trace(cnf_matrix))/tune_y_sample.shape[0]
print (error_rate)

"""
3-2. Classify with SVM
Train on CCA-transformed data
"""

# train on CCA features
Z_train_cca_x, Z_tune1_cca_x,Z_train1_cca_y,Z_tune1_cca_y = train_test_split(z_cca_train,train_y_sample,
                                                                     train_size=.75,random_state=1)

cca_best_error_tune,cca_pred = linear_svm(Z_train_cca_x, Z_train1_cca_y, 
                                           Z_tune1_cca_x, Z_tune1_cca_y, z_cca_tune,
                                            cc=[0.05, 0.1, 0.15])

cnf_matrix = confusion_matrix(tune_y_sample, cca_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True)
plt.show()
#Find accuracy when SVM applied onto development set
error_rate = 1 - (np.matrix.trace(cnf_matrix))/tune_y_sample.shape[0]
print ("Error Rate: ", error_rate)

"""
3-2. Repeat above for different values of dimensions and regularization coefficients
First fix regularization = 1,1, and consider no. of dimensions = 5,100
Next, fix dimension=50, and consider r = 0.1, 10
"""
"""
dimension = 5
"""
n_components, r1, r2 = 5, 1, 1
cca_a = CCA(n_components, r1, r2)
# train
cca_a.fit(train_x_sample1, train_x_sample2)
# transform
z_cca_train = cca_a.transform(train_x_sample1)
z_cca_tune = cca_a.transform(tune_x_sample1)
# train on CCA features
Z_train_cca_x, Z_tune1_cca_x,Z_train1_cca_y,Z_tune1_cca_y = train_test_split(z_cca_train,train_y_sample,
                                                                     train_size=.75,random_state=1)
cca_best_error_tune,cca_pred = linear_svm(Z_train_cca_x, Z_train1_cca_y, 
                                           Z_tune1_cca_x, Z_tune1_cca_y, z_cca_tune,
                                            cc=[0.1, 0.5, 1])

cnf_matrix = confusion_matrix(tune_y_sample, cca_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True)
plt.show()
#Find accuracy when SVM applied onto development set
error_rate = 1 - (np.matrix.trace(cnf_matrix))/tune_y_sample.shape[0]
print ("Error Rate: ", error_rate)


"""
dimension = 200
"""
n_components, r1, r2 = 200, 1, 1
cca_a = CCA(n_components, r1, r2)
# train
cca_a.fit(train_x_sample1, train_x_sample2)
# transform
z_cca_train = cca_a.transform(train_x_sample1)
z_cca_tune = cca_a.transform(tune_x_sample1)
# train on CCA features
Z_train_cca_x, Z_tune1_cca_x,Z_train1_cca_y,Z_tune1_cca_y = train_test_split(z_cca_train,train_y_sample,
                                                                     train_size=.75,random_state=1)
cca_best_error_tune,cca_pred = linear_svm(Z_train_cca_x, Z_train1_cca_y, 
                                           Z_tune1_cca_x, Z_tune1_cca_y, z_cca_tune,
                                            cc=[0.01, 0.05, 0.1])

cnf_matrix = confusion_matrix(tune_y_sample, cca_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True)
plt.show()
#Find accuracy when SVM applied onto development set
error_rate = 1 - (np.matrix.trace(cnf_matrix))/tune_y_sample.shape[0]
print ("Error Rate: ", error_rate)

"""
dimension = 50, r = 1
"""
n_components, r1, r2 = 50, 1, 1
cca_a = CCA(n_components, r1, r2)
# train
cca_a.fit(train_x_sample1, train_x_sample2)
# transform
z_cca_train = cca_a.transform(train_x_sample1)
z_cca_tune = cca_a.transform(tune_x_sample1)

#Visualize
tune_x_image_hat = cca_a.reconstruct(z_cca_tune).reshape((-1, 28, 28)).transpose((0, 2, 1))
ax = plot_images(tune_x_image_hat[::200], 4, 4, 28, 28)
plt.show()

# train on CCA features
Z_train_cca_x, Z_tune1_cca_x,Z_train1_cca_y,Z_tune1_cca_y = train_test_split(z_cca_train,train_y_sample,
                                                                     train_size=.75,random_state=1)
cca_best_error_tune,cca_pred = linear_svm(Z_train_cca_x, Z_train1_cca_y, 
                                           Z_tune1_cca_x, Z_tune1_cca_y, z_cca_tune,
                                            cc=[0.01, 0.05, 0.1])

cnf_matrix = confusion_matrix(tune_y_sample, cca_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True)
plt.show()
#Find accuracy when SVM applied onto development set
error_rate = 1 - (np.matrix.trace(cnf_matrix))/tune_y_sample.shape[0]
print ("Error Rate: ", error_rate)


"""
dimension = 50, r=0.1, 0.1
"""
n_components, r1, r2 = 50, 0.1, 0.1
cca_a = CCA(n_components, r1, r2)
# train
cca_a.fit(train_x_sample1, train_x_sample2)
# transform
z_cca_train = cca_a.transform(train_x_sample1)
z_cca_tune = cca_a.transform(tune_x_sample1)

tune_x_image_hat = cca_a.reconstruct(z_cca_tune).reshape((-1, 28, 28)).transpose((0, 2, 1))
ax = plot_images(tune_x_image_hat[::200], 4, 4, 28, 28)
plt.show()

# train on CCA features
Z_train_cca_x, Z_tune1_cca_x,Z_train1_cca_y,Z_tune1_cca_y = train_test_split(z_cca_train,train_y_sample,
                                                                     train_size=.75,random_state=1)
cca_best_error_tune,cca_pred = linear_svm(Z_train_cca_x, Z_train1_cca_y, 
                                           Z_tune1_cca_x, Z_tune1_cca_y, z_cca_tune,
                                            cc=[0.01, 0.1, 0.5, 1])

cnf_matrix = confusion_matrix(tune_y_sample, cca_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True)
plt.show()
#Find accuracy when SVM applied onto development set
error_rate = 1 - (np.matrix.trace(cnf_matrix))/tune_y_sample.shape[0]
print (error_rate)


"""
dimension = 50, r=10, 10
"""
n_components, r1, r2 = 50, 10, 10
cca_a = CCA(n_components, r1, r2)
# train
cca_a.fit(train_x_sample1, train_x_sample2)
# transform
z_cca_train = cca_a.transform(train_x_sample1)
z_cca_tune = cca_a.transform(tune_x_sample1)

tune_x_image_hat = cca_a.reconstruct(z_cca_tune).reshape((-1, 28, 28)).transpose((0, 2, 1))
ax = plot_images(tune_x_image_hat[::200], 4, 4, 28, 28)
plt.show()

# train on CCA features
Z_train_cca_x, Z_tune1_cca_x,Z_train1_cca_y,Z_tune1_cca_y = train_test_split(z_cca_train,train_y_sample,
                                                                     train_size=.75,random_state=1)
cca_best_error_tune,cca_pred = linear_svm(Z_train_cca_x, Z_train1_cca_y, 
                                           Z_tune1_cca_x, Z_tune1_cca_y, z_cca_tune,
                                            cc=[0.01, 0.1, 0.5, 1])
cnf_matrix = confusion_matrix(tune_y_sample, cca_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True)
plt.show()
#Find accuracy when SVM applied onto development set
error_rate = 1 - (np.matrix.trace(cnf_matrix))/tune_y_sample.shape[0]
print ("Error Rate: ", error_rate)




"""
Q4-4. Train with IDX
Data processing
"""

train_x_view1_subset1 = train_x_sample1[subset_1,]
train_x_view2_subset1 = train_x_sample2[subset_1,]
train_x_view1_subset2 = train_x_sample1[subset_2,]
train_x_view2_subset2 = train_x_sample2[subset_2,]
train_y_subset1 = train_y_sample[subset_1,]
train_y_subset2 = train_y_sample[subset_2,]
train_x_view1_idx = train_x_sample1[subset_both_comp,]
train_x_view2_idx = train_x_sample2[subset_both_comp,]

#First, use remaining data for learning
cca_idx = CCA(20,1,1)
# train
cca_idx.fit(train_x_view1_idx, train_x_view2_idx)

# transform development set into PCA projection
z_idx_cca_tune = cca_idx.transform(tune_x_sample1)
z_idx_cca_subset1 = cca_idx.transform(train_x_view1_subset1)
z_idx_cca_subset2 = cca_idx.transform(train_x_view1_subset2)

"""
SVM classifer learning
We use subset 1 for training, and subset 2 for validation/tuning
"""
#PERFORMANCE on non-PCA data - REFER TO PREVIOUS QUESTION


#PERFORM on CCA_transformed data
cca_50_best_error_tune, cca_50_pred = linear_svm(z_idx_cca_subset1, train_y_subset1, 
                                           z_idx_cca_subset2, train_y_subset2, z_idx_cca_tune,
                                           cc=[0.2, 0.6, 1])

cnf_matrix_50 = confusion_matrix(tune_y_sample, cca_50_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix_50, classes=class_names, normalize=True)
plt.show()
error_rate = 1 - (np.matrix.trace(cnf_matrix_50))/tune_y_sample.shape[0]
#Find accuracy when SVM applied onto development set
print (error_rate)

