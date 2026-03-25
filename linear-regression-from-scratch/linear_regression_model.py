import numpy as np 

#----------------Loss Fucntion-----------------#
def loss_function(w, b, points):
    points = np.array(points)
    
    x = points[:, :-1]
    y = points[:, -1]
    
    y_pred = np.dot(x, w) + b
    
    mean_squared_error = np.mean((y_pred-y)**2)
    
    return mean_squared_error

#---------------Gradient Descent Step--------------#
def gradient_descent(w, b, points, L):
    points = np.array(points)
    n = len(points)
    
    x = points[:, :-1]
    y = points[:, -1]
    
    y_pred = np.dot(x, w) + b
    error = y_pred - y
    
    w_gradient = (1/n) * np.dot(x.T, error)
    b_gradient = (1/n) * np.sum(error)
    
    w = w - L * w_gradient
    b = b - L * b_gradient
    
    return w, b

#-------------------Training Script--------------#
def train(points, epochs=1000, L=0.01):
    n_features = points.shape[1]-1
    w = np.zeros(n_features)
    b = 0
    
    losses = []

    for i in range(epochs):
        w, b = gradient_descent(w, b, points, L)
        
        loss = loss_function(w, b ,points)
        
        losses.append(loss)
            
    return w, b, losses

