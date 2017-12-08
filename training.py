from numerical_gradient_checking import *
from scipy import optimize

class trainer(object):
    def __init__(self,N):
        # Make local reference to network
        self.N = N

    def callBackF(self,params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X,self.Y))

    def costFunctionWrapper(self,params,X,Y):
        self.N.setParams(params)
        cost = self.N.costFunction(X,Y)
        grad = self.N.computeGradient(X,Y)

        return cost,grad

    def train(self,X,Y):
        # Make an internal variable for callback function
        self.X = X
        self.Y = Y

        # Make empty list to store cost
        self.J = []

        params0 = self.N.getParams()

        options = {'maxiters':200, 'desp': True}

        _res = optimize.minimize(self.costFunctionWrapper,params0, jac=True,method='BFGS',\
                                 args=(X,Y), options=options, callback=self.callBackF)

        self.N.setParams(_res.X)
        self.optimizationResult = _res


NN = Neaural_Network()

T = trainer(NN)

T.train(X,Y)

NN.costFunctionPrime(X,Y)

NN.forward(X,Y)

# # Test Network for various combination
# hoursSleep = linspace(0, 10, 100)
# hoursStudy = linspace(0, 5, 100)
#
#
# #Normalize data (same way training data way normalized)
# hoursSleepNorm = hoursSleep/10.
# hoursStudyNorm = hoursStudy/5.
