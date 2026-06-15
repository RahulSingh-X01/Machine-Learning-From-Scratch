import pandas as pd
import numpy as np
from perceptron_model import train
from perceptron_model import forward

# Data loading
data = pd.read_csv(r"C:\Users\rahul\Documents\docs\Github Projects\ML from scratch\Machine-Learning-From-Scratch\perceptron-from-scratch\perceptron_dataset.csv")

# Spliting the data for training and testing purpose
points = data.values

# Shuffle
np.random.shuffle(points)

# Split
split = int(0.8 * len(points))
train_data = points[:split]
test_data = points[split:]

# Training model 
w, b = train(100, train_data)

# Prediction
X = test_data[:, :-1]
y = test_data[:, -1]
correct = 0
for i in range(X.shape[0]):
    
    y_pred = forward(w, b, X[i])
    if y_pred == y[i]:
        correct += 1

# Measuring accuracy 
accuracy = (correct/len(test_data)) * 100

print(f"Accurracy on test data: {accuracy:.2f} %")