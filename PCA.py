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
PCA - DATA PROCESSING
"""
np.random.seed(1)

datapath="data/noisy_distribute.mat"
print("Datapath is: %s" % datapath)

trainData,tuneData,testData = read_mnist(datapath)
#test_x_sample = testData
#test_x_image = np.reshape(test_x_sample, [test_x_sample.shape[0],28,28]).transpose(0, 2, 1)
train_x_sample = trainData.images
train_x_image = np.reshape(train_x_sample, [train_x_sample.shape[0],28,28]).transpose(0, 2, 1)
train_y_sample = np.reshape(trainData.labels, [train_x_sample.shape[0]])
tune_x_sample = tuneData.images
tune_x_image = np.reshape(tune_x_sample, [tune_x_sample.shape[0],28,28]).transpose(0, 2, 1)
tune_y_sample = np.reshape(tuneData.labels, [tune_x_sample.shape[0]])

"""
Visualize a few input digits
"""
ax = plot_images(train_x_image[::1001], 2, 2, 28, 28)
plt.show()

"""
Visualize more input digits, using smaller scale
"""
train_x_rescale = resize(train_x_image,10,10)
ax = plot_images(train_x_rescale[::100], 10, 10, 10, 10)
plt.show()


"""
Q2-2. Implement PCA solution
"""

class PCA(object):
    def __init__(self, n_components):
        """
        Set up PCA
        :in: n_components: number of components to keep
        """
        self.n_components = n_components

    def fit(self, X):
        """
        TODO: fit the model with training data
        :in: X, 2d-array of shape (n_samples, n_features): data matrix
        """
        n_features = X.shape[1]
        self.w1 = np.zeros((self.n_components, n_features))
        self.mu = np.zeros(n_features)
        self.mu = X.mean(axis=0)
        scaled_X = X - self.mu
        cov_mat = np.cov(scaled_X.T)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
        for j in range(self.n_components):
            ranked_eig_val, ranked_eig_vecs = eig_pairs[j]
            self.w1[j,] = ranked_eig_vecs
        self.W = self.w1.T
        return
  
    def transform(self, X):
        """
        TODO: apply PCA on X
        --------
        :in: X, 2d-array of shape (n_samples, n_features): data matrix
        :out:Z, 2d-array of shape (n_samples, n_components): feature matrix
        """
        scaled_X = X - self.mu
        self.Z = scaled_X @ self.W
        return self.Z
    
    def reconstruct(self, Z):
        """
        TODO: transform feature Z back to its original space
        --------
        :in: Z, 2d-array of shape (n_samples, n_components): feature matrix
        :out: X_hat, 2d-array of shape (n_samples, n_features): reconstructed data matrix
        """
        wz = Z @ self.W.T
        self.X_hat = self.mu + wz
        return self.X_hat

"""TEST CASES
#Create some test matrices
test1 = np.array([[1,1,0,3,1, 2], [ 1,2,2,1,3, 2], [ 2,3,3,3,3, 4],
                 [ 4,4,5,4,5, 3], [ 5,5,6,4,6, 5]])
test2 = np.array([[ 1,2,0,0,1, 2], [ 0,2,3,1,3,2], [ 2,2,3,3,3,4],
                 [ 5,4,3,4,5, 3], [ 4,5,4,6,5,5]])

pcatest = PCA(2)
pcatest.fit(test1)
ztest = pcatest.transform(test2)
pcatest.reconstruct(ztest)
"""

n_components = 20

pca = PCA(n_components)

# train
pca.fit(train_x_sample)

# transform
z_pca_train = pca.transform(train_x_sample)
z_pca_tune = pca.transform(tune_x_sample)

"""
Visualize results
"""
# reconstruct image at k=20
tune_x_image_hat = pca.reconstruct(z_pca_tune).reshape((-1, 28, 28)).transpose((0, 2, 1))
ax = plot_images(tune_x_image_hat[::500], 4, 4, 28, 28)
plt.show()
#Compare with previous images
tune_x = tune_x_sample.reshape((-1, 28, 28)).transpose((0, 2, 1))
ax = plot_images(tune_x[::500], 4, 4, 28, 28)
plt.show()

# Visualization without annotation
vis_sample_rate = 10
plot_embedding(z_pca_tune[::vis_sample_rate, :2],tune_y_sample[::vis_sample_rate])
plt.show()

# Visualization with annotation
tune_x_rescale = resize(tune_x_image,10,10)
plot_embedding_annotation(z_pca_tune[::vis_sample_rate, :2],tune_x_rescale[::vis_sample_rate],0.001)
plt.show()


"""
Q2-3. Classify with SVM
"""
#Split training set data into SVM training and SVM tuning set
svm_train1_x,svm_tune1_x,svm_train1_y,svm_tune1_y = train_test_split(train_x_sample,train_y_sample,
                                                                     train_size=.75,random_state=1)

# train on raw image, prediction on development set. Time: Took quite a while :(
#Note numbers in cc are selected after some trials
raw_best_error_tune, raw_pred = linear_svm(svm_train1_x, svm_train1_y, 
                                           svm_tune1_x, svm_tune1_y, tune_x_sample,
                                           cc=[0.06, 0.075, 0.09])

#Set up confusion matrix
cnf_matrix = confusion_matrix(tune_y_sample, raw_pred)

class_names = np.array([0,1,2,3,4,5,6,7,8,9])

#Plot figure
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True)
plt.show()
error_rate = 1 - (np.matrix.trace(cnf_matrix))/tune_y_sample.shape[0]
#Find accuracy when SVM applied onto development set
print (error_rate)

# train on PCA features
Z_train1_x, Z_tune1_x,Z_train1_y,Z_tune1_y = train_test_split(z_pca_train,train_y_sample,
                                                                     train_size=.75,random_state=1)

pca_best_error_tune, pca_pred = linear_svm(Z_train1_x, Z_train1_y, 
                                           Z_tune1_x, Z_tune1_y, z_pca_tune,
                                            cc=[0.075, 0.1, 0.2])

class_names = np.array([0,1,2,3,4,5,6,7,8,9])
cnf_matrix = confusion_matrix(tune_y_sample, pca_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True)
plt.show()
#Find accuracy when SVM applied onto development set
error_rate = 1 - (np.matrix.trace(cnf_matrix))/tune_y_sample.shape[0]
print (error_rate)


"""
Q2-4. Train with IDX
Data processing
"""
idx = sio.loadmat("data\idx.mat")
subset_1 = idx["p1"]
subset_2 = idx["p2"]
subset_both = np.concatenate((subset_1,subset_2),axis=1)
subset_1 = np.reshape(subset_1, (50,))
subset_2 = np.reshape(subset_2, (50,))
l = [i for i in range(50000)]

#Find set of indices not in labelled data 1 and 2
subset_both_comp = np.array([x for x in l if x not in subset_both])

train_x_subset1 = train_x_sample[subset_1,]
train_x_subset2 = train_x_sample[subset_2,]
train_y_subset1 = train_y_sample[subset_1,]
train_y_subset2 = train_y_sample[subset_2,]
train_x_idx = train_x_sample[subset_both_comp,]


#First, use remaining data for learning
train_x_idx = train_x_sample[subset_both_comp,]

n_components=20
pca = PCA(n_components)
# train
pca.fit(train_x_idx)

# transform development set into PCA projection
z_idx_pca_tune = pca.transform(tune_x_sample)
z_idx_pca_subset1 = pca.transform(train_x_subset1)
z_idx_pca_subset2 = pca.transform(train_x_subset2)

"""
SVM classifer learning
We use subset 1 for training, and subset 2 for validation/tuning
"""
#FIRST PERFORM on non-PCA data
raw_50_best_error_tune, raw_50_pred = linear_svm(train_x_subset1, train_y_subset1, 
                                           train_x_subset2, train_y_subset2, tune_x_sample,
                                           cc=[0.05, 0.1, 0.5])

cnf_matrix_50 = confusion_matrix(tune_y_sample, raw_50_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix_50, classes=class_names, normalize=True)
plt.show()
error_rate = 1 - (np.matrix.trace(cnf_matrix_50))/tune_y_sample.shape[0]
#Find accuracy when SVM applied onto development set
print (error_rate)


#PERFORM on PCA_transformed data
pca_50_best_error_tune, pca_50_pred = linear_svm(z_idx_pca_subset1, train_y_subset1, 
                                           z_idx_pca_subset2, train_y_subset2, z_idx_pca_tune,
                                           cc=[0.05, 0.1, 0.5])

cnf_matrix_50 = confusion_matrix(tune_y_sample, pca_50_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix_50, classes=class_names, normalize=True)
plt.show()
error_rate = 1 - (np.matrix.trace(cnf_matrix_50))/tune_y_sample.shape[0]
#Find accuracy when SVM applied onto development set
print (error_rate)



"""
SVM Classifier Learning
APPLY ON ORIGINAL DATASET
"""
datapath="data/original_distribute.mat"
print("Datapath is: %s" % datapath)

c_trainData,c_tuneData,c_testData = read_mnist(datapath)
c_train_x_sample = c_trainData.images
c_train_x_image = np.reshape(c_train_x_sample, [c_train_x_sample.shape[0],28,28]).transpose(0, 2, 1)
c_train_y_sample = np.reshape(c_trainData.labels, [c_train_x_sample.shape[0]])
c_tune_x_sample = c_tuneData.images
c_tune_x_image = np.reshape(c_tune_x_sample, [c_tune_x_sample.shape[0],28,28]).transpose(0, 2, 1)
c_tune_y_sample = np.reshape(c_tuneData.labels, [c_tune_x_sample.shape[0]])

"""
Visualize more input digits, using smaller scale
"""
c_train_x_rescale = resize(c_train_x_image,10,10)
ax = plot_images(c_train_x_rescale[::100], 10, 10, 10, 10)
plt.show()

c_train_x_subset1 = c_train_x_sample[subset_1,]
c_train_x_subset2 = c_train_x_sample[subset_2,]
c_train_y_subset1 = c_train_y_sample[subset_1,]
c_train_y_subset2 = c_train_y_sample[subset_2,]
c_train_x_idx = c_train_x_sample[subset_both_comp,]


#First, use remaining data for learning
c_train_x_idx = c_train_x_sample[subset_both_comp,]
c_pca = PCA(n_components)
# train
c_pca.fit(c_train_x_idx)

# transform development set into PCA projection
c_z_idx_pca_tune = c_pca.transform(c_tune_x_sample)
c_z_idx_pca_subset1 = c_pca.transform(c_train_x_subset1)
c_z_idx_pca_subset2 = c_pca.transform(c_train_x_subset2)

"""
CLEAN SVM Classifier Learning
FIRST PERFORM on non-PCA data
"""

c_raw_50_best_error_tune, c_raw_50_pred = linear_svm(c_train_x_subset2, c_train_y_subset2, 
                                           c_train_x_subset1, c_train_y_subset1, c_tune_x_sample,
                                           cc=[0.05, 0.1, 0.5])

c_cnf_matrix_50 = confusion_matrix(c_tune_y_sample, c_raw_50_pred)
plt.figure()
plot_confusion_matrix(c_cnf_matrix_50, classes=class_names, normalize=True)
plt.show()
error_rate = 1 - (np.matrix.trace(c_cnf_matrix_50))/c_tune_y_sample.shape[0]
#Find accuracy when SVM applied onto development set
print (error_rate)


#PERFORM on PCA_transformed data
c_pca_50_best_error_tune, c_pca_50_pred = linear_svm(c_z_idx_pca_subset2, c_train_y_subset2, 
                                           c_z_idx_pca_subset2, c_train_y_subset2, c_z_idx_pca_tune,
                                           cc=[0.05, 0.1, 0.5])

c_cnf_matrix_50 = confusion_matrix(c_tune_y_sample, c_pca_50_pred)
plt.figure()
plot_confusion_matrix(c_cnf_matrix_50, classes=class_names, normalize=True)
plt.show()
error_rate = 1 - (np.matrix.trace(c_cnf_matrix_50))/c_tune_y_sample.shape[0]
#Find accuracy when SVM applied onto development set
print (error_rate)

