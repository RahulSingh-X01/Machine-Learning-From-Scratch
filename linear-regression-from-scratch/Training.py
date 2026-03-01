import pandas as pd
import numpy as np
from linear_regression_model import train, loss_function

# Load data
data = pd.read_csv("dataset.csv")
points = data.values

# Shuffle
np.random.shuffle(points)

# Split
split = int(0.8 * len(points))
train_data = points[:split]
test_data = points[split:]

# Train model
w, b = train(train_data)

# Evaluate
test_loss = loss_function(w, b, test_data)
print("Final Test Loss:", test_loss)

