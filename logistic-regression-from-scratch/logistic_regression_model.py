import numpy as np

#----------------Loss fucntion-------------#
def loss_function(w, b, points):
    points = np.array(points)
    
    x = points[:, :-1]
    y = points[:, -1]
    
    y_pred = 1/(1+np.exp(-(np.dot(x, w)+b)))
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
    
    error = (y*np.log(y_pred))+((1-y)*np.log(1-y_pred))
    
    binary_cross_entropy = -np.mean(error)
    
    return binary_cross_entropy




