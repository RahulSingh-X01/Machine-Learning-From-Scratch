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

#----------------Gradeint Descent------------#
def gradient_desent(w, b, L, points):
    points = np.array(points)
    m = len(points)
    
    x = points[:, :-1]
    y = points[:, -1]
    
    y_pred =  1/(1+np.exp(-(np.dot(x, w)+b)))
    
    error = y_pred - y
    
    dJ_dw = (1/m)*np.dot(x.T, error)
    dJ_db = (1/m)*np.sum(error)
    
    w = w - L * dJ_dw
    b = b - L * dJ_db
    
    return w, b


