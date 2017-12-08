#%pylab inline

from Data_and_Structure import *

print X.shape
print Y.shape

class Neural_Network(object):
    def __init__(self):
        # Define Hyperparameter
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1

        # Weights
        self.W1 = np.random.rand(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.rand(self.hiddenLayerSize,self.outputLayerSize)

    def forward(self,X):
        # Propagate input through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self,z):
        return 1/(1 + np.exp(-z))