import numpy as np

# X = (data) , Y = (output)
# in this case (hour of sleep, hours studying) and (Score on test)

X = np.array([3,5],[5,1],[10,2],dtype=float)
Y = np.array([75],[82],[93],dtype=float)

# Normalize
X = X/np.amax(X, axis=0)
Y = Y/100

