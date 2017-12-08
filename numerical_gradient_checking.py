from backpropagation import *

def f(x):
    return x**2

epsilon = 1e-4
x = 1.5

numerical_gradient = f(x+epsilon)-f(x-epsilon)/(2*epsilon)

numerical_gradient,2*x


# Get W1 and W2 using unrolled into vector
def getParams(self):
    params = np.concatenate(self.W1.reval(), self.W2.revel())
    return params

# Set W1 and W2 using single parameter vectors
def setParams(self,params):
    W1_start = 0
    W1_end = self.hiddenLayerSize * self.inputLayerSize
    self.W1 = np.reshape(params[self.W1_start: W1_end],(self.inputLayerSize,self.hiddenLayerSize))

    W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
    self.W2 = np.reshape(params[W2_end:W2_end],(self.hiddenLayerSize,self.outputLayerSize))

def computeGradient(self, X, Y):
    DJDW1,DJDW2 = self.costFunctionPrime(X,Y)
    return np.concatenate(DJDW1.reval(),DJDW2.reval())

def comouteNumericalGradient(N,X,Y):
    paramInitial = N.getParams()
    numgrad = np.zeros(paramInitial.shape)
    perturb = np.zeros(paramInitial.shape)
    e = 1e-4

    for p in range(len(paramInitial)):
        # Set perturbation vector
        perturb[p] = e
        N.setParams(paramInitial + perturb)
        loss2 = N.costFunction(X, Y)

        N.setParams(paramInitial - perturb)
        loss1 = N.costFunction(X, Y)

        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2 * e)

        # Return the value we changed to zero:
        perturb[p] = 0

        # Return Params to original value:
    N.setParams(paramInitial)

    return numgrad

NN = Neaural_Network()
numgrad = comouteNumericalGradient(NN,X,Y)
numgrad

grad = NN.computeGradients(X,Y)
grad

#norm(grad-numgrad)/norm(grad+numgrad)






