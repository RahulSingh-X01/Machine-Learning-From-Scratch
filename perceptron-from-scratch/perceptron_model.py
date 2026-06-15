import numpy  as np
import random as rd

#----------------Forward Pass--------------#
def forward(w, b, X):
    z = np.dot(X, w) + b
    
    if z >= 0:
        return 1 
    else:
        return 0

#---------------Training loop-------------#
def train(epochs, data, lr=0.01):
    X = data[: , :-1]
    y = data[: , -1]
    w = np.random.rand(X.shape[1])
    b = 0
    
    for epoch in range(epochs):
        errors = 0
        for i in range(X.shape[0]):
            x = X[i]
            target = y[i]
            y_pred = forward(w, b, x)
            error = target - y_pred
            if error != 0:
                w += lr * error * x
                b += lr * error
                
                errors += 1
        
        if errors == 0:
            break
    return w, b
        