import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(losses):
    plt.figure()
    plt.plot(losses)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def plot_actual_vs_predicted(y_true, y_pred):
    plt.figure()
    plt.scatter(y_true, y_pred)
    
    # Diagonal reference line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    
    plt.plot([min_val, max_val], [min_val, max_val], color="red")
    
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.show()
    
def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color="red")
    
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()