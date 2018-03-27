from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
import scipy.stats as sp
import pandas as pd

# Please implement the fit and predict methods of this class. You can add
# additional private methods by beginning them with two underscores. It may
# look like the __dummyPrivateMethod below.  You can feel free to change any of
# the class attributes, as long as you do not change any of the given function
# headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method.
class GaussianGenerativeModel:
    num_classes = 3
    def __init__(self, isSharedCovariance=False):
        self.isSharedCovariance = isSharedCovariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        self.C = []
        self.C.append(self.X[self.Y == 0])
        self.C.append(self.X[self.Y == 1])
        self.C.append(self.X[self.Y == 2])

        self.mu = []
        self.mu.append(np.mean(self.C[0], axis=0))
        self.mu.append(np.mean(self.C[1], axis=0))
        self.mu.append(np.mean(self.C[2], axis=0))

        self.sigma = []

        if self.isSharedCovariance:
            self.sigma.append(np.cov(X.T))
            self.sigma.append(np.cov(X.T))
            self.sigma.append(np.cov(X.T))
        else:
            self.sigma.append(np.cov(self.C[0].T))
            self.sigma.append(np.cov(self.C[1].T))
            self.sigma.append(np.cov(self.C[2].T))

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        Y = []
        Y.append(multivariate_normal.pdf(X_to_predict, mean=self.mu[0], cov=self.sigma[0]))
        Y.append(multivariate_normal.pdf(X_to_predict, mean=self.mu[1], cov=self.sigma[1]))
        Y.append(multivariate_normal.pdf(X_to_predict, mean=self.mu[2], cov=self.sigma[2]))
        predictions = pd.DataFrame({'Class 1': Y[0], 'Class 2': Y[1], 'Class 3': Y[2]})

        return np.argmax(np.array(predictions), axis=1)

    # Do not modify this method!
    def visualize(self, output_file, width=3, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min,
            y_max, .05))

        # Flatten the grid so the values match spec for self.predict
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        X_topredict = np.vstack((xx_flat,yy_flat)).T

        # Get the class predictions
        Y_hat = self.predict(X_topredict)
        Y_hat = Y_hat.reshape((xx.shape[0], xx.shape[1]))

        cMap = c.ListedColormap(['r','b','g'])

        # Visualize them.
        plt.figure()
        plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=self.Y, cmap=cMap, edgecolor='k')
        plt.savefig(output_file)
        if show_charts:
            plt.show()
