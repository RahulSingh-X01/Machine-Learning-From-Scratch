import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

data = pd.read_csv("dataset.csv")

def loss_fucntion(w, b, points):
    points = np.array(points)
    
    x = points[:, :-1]
    y = points[:, -1]
    
    y_pred = np.dot(x, w) + b
    
    mean_sqaured_error = np.mean((y_pred-y)**2)
    
    return mean_sqaured_error