import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

data = pd.read_csv("dataset.csv")

def loss_function(w, b, points):
    points = np.array(points)
    
    x = points[:, :-1]
    y = points[:, -1]
    
    y_pred = np.dot(x, w) + b
    
    mean_squared_error = np.mean((y_pred-y)**2)
    
    return mean_squared_error

def gradient_descent(w, b, points):
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


    


    
        
        
        
        
        