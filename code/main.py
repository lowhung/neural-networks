import sys
import argparse
import os
import numpy as np
import utils
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor as NeuralNet
from sklearn.neural_network import MLPClassifier as Classifier
from sklearn import datasets
from sklearn.datasets import load_boston

if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True, choices=['1', '2'])

    io_args = parser.parse_args()
    question = io_args.question

    if question == '1':

        data = utils.load_dataset('basisData')

        X = data['X']
        y = data['y'].ravel()
        Xtest = data['Xtest']
        ytest = data['ytest'].ravel()
        n,d = X.shape
        t = Xtest.shape[0]

        model = NeuralNet(
                        solver="lbfgs",
                        hidden_layer_sizes=(120,100), alpha=2, activation='logistic', max_iter=4000)
        model.fit(X,y)

        # Comput training error
        yhat = model.predict(X)
        trainError = np.mean((yhat - y)**2)
        print("Training error = ", trainError)

        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.mean((yhat - ytest)**2)
        print("Test error     = ", testError)

        plt.figure()
        plt.plot(X, y, 'b.', label="training data", markersize=2)
        plt.title('Training Data')
        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot()
        plt.plot(Xhat, yhat, 'g', label="neural network")
        plt.ylim([-300,400])
        plt.legend()
        figname = os.path.join("..","figs","basisData.pdf")
        print("Saving", figname)
        plt.savefig(figname)

    elif question == '2':

        data = utils.load_dataset("citiesSmall")
        # diabetes = datasets.load_diabetes()

        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']
        n, d = X.shape

        # Create the model
        model = Classifier(
                        solver="lbfgs",
                        hidden_layer_sizes=(80, 80), alpha=1.2, activation='logistic', max_iter=4000)

        # Train the model using training sets
        model.fit(X,y)

        # Comput training error
        yhat = model.predict(X)
        trainError = np.mean((yhat - y)**2)
        print("Training error = ", trainError)

        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.mean((yhat - ytest)**2)
        print("Test error     = ", testError)

        utils.plotClassifier(model, X, y)
