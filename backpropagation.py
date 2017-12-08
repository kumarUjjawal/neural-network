from forward_propagation import *

def sigmoid(z):
    # Apply sigmoid fuction to scalars,vectors and matrices
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    # Derivative of sigmoid function
    return np.exp(-z)/((1+np.exp(-z)**2))

def costFunctionPrime(self,X,Y):
    # Compute derivative for W and W2 for a given X and Y
    self.yHat = self.forward(X)

    delta3 = np.multiply(-(Y - self.yHat), self.sigmoidPrime(self.z3))
    DJDW2 = np.dot(self.a2.T,delta3)

    delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
    DJDW1 = np.dot(X.T,delta2)

    return DJDW1,DJDW2

NN = Neaural_Network()

cost1 = NN.costFunction(X,Y)

DJDW1,DJDW2 = NN.costFunctionPrime(X,Y)

scalar = 3

scalar = 3
NN.W1 = NN.W1 + scalar*DJDW1
NN.W2 = NN.W2 + scalar*DJDW2
cost2 = NN.costFunction(X,Y)

print cost1,cost2

dJdW1, dJdW2 = NN.costFunctionPrime(X,Y)
NN.W1 = NN.W1 - scalar*DJDW1
NN.W2 = NN.W2 - scalar*DJDW2
cost3 = NN.costFunction(X, Y)

print cost1,cost2

