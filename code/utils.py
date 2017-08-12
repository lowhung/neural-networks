import os.path
import numpy as np
from numpy.linalg import norm
import pylab as plt
import pickle
import sys
import scipy.sparse

DATA_DIR = 'data'
FIGS_DIR = 'figs'

def savefig(fname, verbose=True):
    path = os.path.join('..', FIGS_DIR, fname)
    plt.savefig(path)
    if verbose:
        print("\nFigure saved as '{}'".format(path))

def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.

    return (X - mu) / sigma

def load_dataset(dataset_name):
    """Loads the dataset corresponding to the dataset name

    Parameters
    ----------
    dataset_name : name of the dataset

    Returns
    -------
    data :
        Returns the dataset as 'dict'
    """

    return load_pkl(os.path.join('..',DATA_DIR,'{}.pkl'.format(dataset_name)))

def load_pkl(fname):
    """Reads a pkl file.

    Parameters
    ----------
    fname : the name of the .pkl file

    Returns
    -------
    data :
        Returns the .pkl file as a 'dict'
    """
    if not os.path.isfile(fname):
        raise ValueError('File {} does not exist.'.format(fname))

    if sys.version_info[0] < 3:
        # Python 2
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    else:
        # Python 3
        with open(fname, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

    return data

def plotClassifier(model, X, y):
    """plots the decision boundary of the model and the scatterpoints
       of the target values 'y'.

    Assumptions
    -----------
    y : it should contain two classes: '1' and '2'

    Parameters
    ----------
    model : the trained model which has the predict function

    X : the N by D feature array

    y : the N element vector corresponding to the target values

    """
    x1 = X[:, 0]
    x2 = X[:, 1]

    x1_min, x1_max = int(x1.min()) - 1, int(x1.max()) + 1
    x2_min, x2_max = int(x2.min()) - 1, int(x2.max()) + 1

    x1_line =  np.linspace(x1_min, x1_max, 200)
    x2_line =  np.linspace(x2_min, x2_max, 200)

    x1_mesh, x2_mesh = np.meshgrid(x1_line, x2_line)

    mesh_data = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]

    y_pred = model.predict(mesh_data)
    y_pred = np.reshape(y_pred, x1_mesh.shape)

    plt.xlim([x1_mesh.min(), x1_mesh.max()])
    plt.ylim([x2_mesh.min(), x2_mesh.max()])

    plt.contourf(x1_mesh, x2_mesh, -y_pred,
                cmap=plt.cm.RdBu, label="decision boundary",
                alpha=0.6)

    plt.scatter(x1[y==1], x2[y==1], color="b", label="class 1")
    plt.scatter(x1[y==2], x2[y==2], color="r", label="class 2")
    plt.legend()
    plt.title("Model outputs '1' for blue region\n"
              "Model outputs '2' for red region\n""Using a Neural Network")
    fname = os.path.join("..", "figs", "q2plot.pdf")
    plt.savefig(fname)
