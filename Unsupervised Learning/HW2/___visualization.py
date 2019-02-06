from __future__ import print_function
from __future__ import division

#%matplotlib inline
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn import decomposition
from numpy.random import permutation as rpm
from plot import plot_images, plot_embedding, plot_embedding_annotation
from dataset import read_mnist, read_mnist_twoview
from utils import resize, x2p, JacobOptimizer, gen_solution
from svm import linear_svm
from vae import VAE
from ptsne import PTSNE

##Additional packages used for visualization
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap

## TSNE
EPS=1.0e-12
torch.manual_seed(1)

class LowEmb(nn.Module):
    """
    Module on computing t-SNE low-dim embedding
    """
    def __init__(self, n, d):
        """
        Set up
        ------
        :in:
        n: int, number of data
        d: int, dimension to reduce to
        """
        super(LowEmb, self).__init__()
        self.Y = nn.Embedding(n, d)


    def forward(self, P, dof, metric):
        """
        Compute divergence between distribution in low and high dimensional space
        TODO: define your own divergence metric
        ------
        :in:
        P: 2d-array of shape (n_data, n_data), distribution in high-dim data space
        dof: float, degree of freedom of Student-t distribution
        metric: string, type of divergence
        """

        # Compute low-dim distribution Q
        Y = self.Y.weight
        Y2 = torch.sum(Y**2, dim=1, keepdim=True)
        D = Y2 - 2*Y.matmul(Y.t()) + Y2.t()
        Q = (1 - torch.eye(Y.shape[0]).type_as(D))*(1 + D/dof).pow(-(dof + 1)/2.0)
        Q = (Q/Q.sum()).clamp_(min=EPS)

        # Compute discrepency between P and Q
        if metric == 'kl':
            C = torch.sum(P*torch.log(P/Q))
        elif metric == 'reverse_kl':
            C = torch.sum(Q*torch.log(Q/P))
        else:
            #Define Jensen–Shannon divergence
            M = (P+Q)/2
            A = torch.sum(P*torch.log(P/M))
            B = torch.sum(Q*torch.log(Q/M))
            C = 0.5*A + 0.5*B
        return C


class TSNE(object):
    """
    Main Module of t-SNE
    Parameters:
    ------ 
    n_components: int, dimension to reduce to (default: 2)
    perplexity: float, perplexity of data in high-dimensional space (default: 30)
    n_iters: int, number of iterations for optimization (default: 1000)
    lr: float, learning rate (default: 100)
    opt_type: string, type of optimizer
        "Jacob" | "Adam" | "SGD" | "RMSprop" | "Adagrad" | "Adadelta" (default: "Jacob")
    cuda: bool, whether to use GPU if available (default: True)
    dof: float, degree of freedom of Student-t distribution (default: 1.0)
    metric: string, type of divergence between high and low dimensional space (default: "kl")
    early_exaggeration: float, constant to multiply to high dimensional probability,
        used in early stages of optimization to exaggerate the divergence 
        between low and high dimensional space. 
        See original t-SNE paper for details (default: 4)
    early_iters: int, number of iterations to do early exaggeration (default: 100)
    log_interval: int, iteration interval to show error (default: 100)
    init: string, initialization of low-dimensional embedding. 
        Possible options are "random" | "pca" (default: "pca")

    """
    def __init__(self, n_components=2, perplexity=30.0, n_iters=1000, lr=1.0e2, opt_type='Jacob', cuda=True, dof=1.0,
                 metric='kl', early_exaggeration=4., early_iters=100, log_interval=100, init='pca'):

        self.n_iters = n_iters
        self.lr = lr
        self.opt_type = opt_type
        self.n_components = n_components
        self.perplexity = perplexity
        self.device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
        self.dof = dof
        self.metric = metric
        self.early_exaggeration = early_exaggeration
        self.early_iters = early_iters
        self.log_interval = log_interval
        self.init = init

    def fit_transform(self, X):
        """
        Run t-SNE on X
        ------
        :in:
        X: 2d array of shape (n_data, n_dim)
        :out:
        Y: 2d array of shape (n_data, n_components)
        """
        print("Metric: %s" % (self.metric))

        # Compute high-dim distribution matrix P
        P = x2p(X, perplexity=self.perplexity)
        P = (P + P.T)/(2*P.sum())

        # early exageration, P *= Const
        P = P if self.early_exaggeration is None else P * self.early_exaggeration
        P = torch.from_numpy(P.clip(EPS)).type(torch.float)

        lowEmb = LowEmb(X.shape[0], self.n_components)
        
        # PCA intialization (optional)
        if self.init == 'pca':
            lowEmb.Y.weight.data.copy_(torch.from_numpy(self.pca(X)))

        lowEmb.to(self.device)
        P = P.to(self.device)
        if self.opt_type == 'Jacob':
            optimizer = JacobOptimizer(lowEmb.parameters(), lr=self.lr)
        elif hasattr(torch.optim, self.opt_type):
            optimizer = getattr(torch.optim, self.opt_type)(lowEmb.parameters(), lr=self.lr)
        else:
            raise AttributeError("Valid options for opt_type are [Jacob, Adam, SGD, RMSprop, Adagrad, Adadelta]!")

        # Optimization
        for iter in range(self.n_iters):

            lowEmb.zero_grad()
            l = lowEmb(P, dof=self.dof, metric=self.metric)
            l.backward()

            optimizer.step()

            if (iter + 1) % self.log_interval == 0:
                print("Iteration %d, error: %f" % (iter + 1, l))

            # Stop early exaggeration, use true P value
            if iter == self.early_iters and self.early_exaggeration is not None:
                P = P/self.early_exaggeration

        Y = lowEmb.Y.weight.cpu().detach().numpy()
        return Y

    def pca(self, X):
        """
        Perform PCA on X, as optional preprocessing step
        ------
        :in:
        X: 2d array of shape (n_data, n_dim)
        :out:
        Y: 2d array of shape (n_data, n_components)
        """
        Y = decomposition.PCA(n_components=self.n_components).fit_transform(X).astype(np.float32)
        return Y
        
        

np.random.seed(0)

datapath="./data/original_distribute.mat"
print("Data path is: %s" % datapath)

trainData,tuneData,testData=read_mnist(datapath)
test_x_sample = testData
test_x_image = np.reshape(test_x_sample, [test_x_sample.shape[0],28,28]).transpose(0, 2, 1)
train_x_sample = trainData.images
train_x_image = np.reshape(train_x_sample, [train_x_sample.shape[0],28,28]).transpose(0, 2, 1)
train_y_sample = np.reshape(trainData.labels, [train_x_sample.shape[0]])
tune_x_sample = tuneData.images
tune_x_image = np.reshape(tune_x_sample, [tune_x_sample.shape[0],28,28]).transpose(0, 2, 1)
tune_y_sample = np.reshape(tuneData.labels, [tune_x_sample.shape[0]])


"""
Visualize a few input digits
plot_images: Plot a list of selected digits
"""
plot_images(train_x_image[::100], 5, 5, 28, 28)

"""
Visualize more input digits, using smaller scale
"""
train_x_rescale = resize(train_x_image,10,10)
ax = plot_images(train_x_rescale[::100], 10, 10, 10, 10)
plt.show()

"""
Visualize 20% dev set using t-SNE
"""

###FIRST VARY PERPLEXITY
size_tune = 5
for i in [3,10,30,70,100,300,1000]:
    print("Start TSNE!")
    z_tsne_tune = TSNE(n_components=2, perplexity=i, n_iters=500, cuda=True, dof=1.0, 
                       metric="kl", early_exaggeration=4, init='pca').fit_transform(tune_x_sample[::size_tune])
    plot_embedding(z_tsne_tune, tune_y_sample[::size_tune])
#    plot_embedding_annotation(z_tsne_tune, resize(tune_x_image[::size_tune],10,10), 0.001)
    plt.show()

    
"""
Fix perplexity=30
Test new divergence
"""
#Jensen–Shannon_divergence
#Fix perp=30
print("Start TSNE! JS Divergence, Perp=30")
z_tsne_tune_js = TSNE(n_components=2, perplexity=30, n_iters=500, cuda=True, dof=1.0, 
                   metric="others", early_exaggeration=4, init='pca').fit_transform(tune_x_sample[::size_tune])
plot_embedding(z_tsne_tune_js, tune_y_sample[::size_tune])
plot_embedding_annotation(z_tsne_tune_js, resize(tune_x_image[::size_tune],10,10), 0.001)
plt.show()
#Fix perp=70
print("Start TSNE! JS Divergence, Perp=100")
z_tsne_tune_js = TSNE(n_components=2, perplexity=100, n_iters=500, cuda=True, dof=1.0, 
                   metric="others", early_exaggeration=4, init='pca').fit_transform(tune_x_sample[::size_tune])
plot_embedding(z_tsne_tune_js, tune_y_sample[::size_tune])
plot_embedding_annotation(z_tsne_tune_js, resize(tune_x_image[::size_tune],10,10), 0.001)
plt.show()


#Inverse of KL (swop P and Q)
#Fix perp = 30
print("Start TSNE! perp=30")
z_tsne_tune_js = TSNE(n_components=2, perplexity=30, n_iters=500, cuda=True, dof=1.0, 
                   metric="reverse_kl", early_exaggeration=4, init='pca').fit_transform(tune_x_sample[::size_tune])
plot_embedding(z_tsne_tune_js, tune_y_sample[::size_tune])
plot_embedding_annotation(z_tsne_tune_js, resize(tune_x_image[::size_tune],10,10), 0.001)
plt.show()
#Fix perp = 100
print("Start TSNE! perp=100")
z_tsne_tune_js = TSNE(n_components=2, perplexity=100, n_iters=500, cuda=True, dof=1.0, 
                   metric="reverse_kl", early_exaggeration=4, init='pca').fit_transform(tune_x_sample[::size_tune])
plot_embedding(z_tsne_tune_js, tune_y_sample[::size_tune])
plot_embedding_annotation(z_tsne_tune_js, resize(tune_x_image[::size_tune],10,10), 0.001)
plt.show()

    
"""
Vary degrees of freedom in t-distribution
Fix perplexity=30
"""
for i in [5,50,500,5000,50000]:
    print("Start TSNE! ", i)
    z_tsne_tune = TSNE(n_components=2, perplexity=30, n_iters=500, cuda=True, dof= i, 
                       metric="kl", early_exaggeration=4, init='pca').fit_transform(tune_x_sample[::size_tune])
    plot_embedding(z_tsne_tune, tune_y_sample[::size_tune])
    plot_embedding_annotation(z_tsne_tune, resize(tune_x_image[::size_tune],10,10), 0.001)
    plt.show()

    
"""
Visualize 20% dev set using other dimensionality reduction methods
Choose Kernel PCA - Use Gaussian/rbf kernel

gamma : float, default=1/n_features
"""

def my_visualize_emb(train_data, gamma_input):
    """
    :in: train_data: 2d array of shape (n_data, n_dim), training data of my method
    :out: train_features: 2d array of shape (n_data, 2), 2-d feature of training data
    """
    #Perform Kernel PCA
    transformer = KernelPCA(n_components=2, kernel='sigmoid', gamma=gamma_input)
    transformed_data = transformer.fit_transform(train_data)
    return transformed_data

#gamma_list = list(np.arange(0.001, 0.01, 0.0005))
gamma_list = [0.0095, 0.0085, 0.0075, 0.0055, 0.004, 0.0035]
               
for gamma in gamma_list:
    print("Start My Method! Gamma = ", gamma)
    z_my_tune = my_visualize_emb(tune_x_sample[::5], gamma)
    plot_embedding(z_my_tune, tune_y_sample[::5])
#    plot_embedding_annotation(z_my_tune, resize(tune_x_image[::5],10,10), 0.001)
    plt.show()

    
"""
Visualize 20% dev set using other dimensionality reduction methods
Choose IsoMap
hyperparameter: n = number of neighbors
"""

def my_visualize_Isomap(train_data, neighbors):
    #Perform Isomap
    transformer = Isomap(n_components=5, eigen_solver='dense',
                         n_neighbors=neighbors)
    transformed_data = transformer.fit_transform(train_data)
    return transformed_data

neighbor_list = [5,15,30,50,75]
               
for n in neighbor_list:
    print("Start My Method! Nearest Neighbors = ", n)
    z_my_tune = my_visualize_Isomap(tune_x_sample[::5], n)
    plot_embedding(z_my_tune, tune_y_sample[::5])
#    plot_embedding_annotation(z_my_tune, resize(tune_x_image[::5],10,10), 0.001)
    plt.show()

    

"""
Q3.5
Visualize 20% dev set using training on a larger training set
Since this should be parametric, we will use parmetric t-SNE.
Keep perp within the range of 30 for easy comparison. 
Repeat to make sure results are not perp-dependent
"""
training_1 = train_x_sample[::10]
tuning_1 = train_x_sample[::30]

for perp in [30, 70, 100]:
    ptsne = PTSNE(n_inputs=784, n_components=2, batch_size=5000, cuda=True, perplexity=perp, dof=1)
    ptsne.fit(training_1, tuning_1, epochs=10)

    z_separate = ptsne.transform(tune_x_sample[::5])
    plot_embedding(z_separate, tune_y_sample[::5])
    plot_embedding_annotation(z_separate, resize(tune_x_image[::5],10,10), 0.001)
    print(perp)
    plt.show()

