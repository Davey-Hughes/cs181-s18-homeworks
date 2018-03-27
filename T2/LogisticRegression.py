import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
from scipy.misc import logsumexp

# Please implement the fit and predict methods of this class. You can add
# additional private methods by beginning them with two underscores. It may
# look like the __dummyPrivateMethod below.  You can feel free to change any of
# the class attributes, as long as you do not change any of the given function
# headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method.
class LogisticRegression:
    num_classes = 3

    def __init__(self, eta, lambda_parameter):
        self.eta = eta
        self.lambda_parameter = lambda_parameter

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def __one_hot(self, Y):
        arr = []
        for y in Y:
            inner = np.zeros(self.num_classes)
            inner[y] = 1
            arr.append(inner)

        return np.array(arr)

    def __softmax(self, X, W):
        xs = np.dot(W, X.T)
        e_x = np.exp(xs - np.max(xs))
        ret = e_x / np.sum(e_x)
        return ret

    def __gradient_loss(self, j):
        acc = np.zeros(self.X.shape[1])
        for i, y in enumerate(self.Y):
            acc += (self.__softmax(self.X[i].T, self.W)[j] - y[j]) * self.X[i]

        return acc

    def __iter(self, j):
        loss = self.__gradient_loss(j)
        self.loss_iters.append(loss)
        self.W[j] = self.W[j] - self.eta * loss - 2 * self.eta * self.lambda_parameter * self.W[j]

    def fit(self, X, C):
        self.X = X
        self.C = C
        self.loss_iters = []

        np.append(X, np.ones((X.shape[0],1)), axis=1)

        self.Y = self.__one_hot(C)
        self.W = np.array([[0.7, 0.4], [0.3, 0.9], [0.2, 0.7]])

        for i in range(10000):
            for j in range(self.num_classes):
                self.__iter(j)

        xs1 = []
        xs2 = []
        for i in range(len(self.loss_iters)):
            xs1.append(self.loss_iters[i][0])
            xs2.append(self.loss_iters[i][1])

        plt.scatter([i + 1 for i in range(len(self.loss_iters))], xs1, s=5)
        # plt.scatter([i + 1 for i in range(len(self.loss_iters))], xs2, s=5)
        plt.savefig('loss_per_iter.png')

        return

    def predict(self, X_to_predict):
        preds = []

        for i in range(len(X_to_predict)):
            preds.append(np.argmax(self.__softmax(X_to_predict[i], self.W)))

        return np.array(preds)

    def visualize(self, output_file, width=2, show_charts=False):
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
        plt.scatter(X[:, 0], X[:, 1], c=self.C, cmap=cMap, edgecolor='k')
        plt.savefig(output_file)
        if show_charts:
            plt.show()
