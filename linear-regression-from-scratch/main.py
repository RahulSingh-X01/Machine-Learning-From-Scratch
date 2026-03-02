import pandas as pd
import numpy as np
from linear_regression_model import train, loss_function
from Visualization import plot_loss_curve, plot_actual_vs_predicted, plot_residuals

# Load data
data = pd.read_csv(r"Machine-Learning-From-Scratch\linear-regression-from-scratch\dataset.csv")
points = data.values

# Shuffle
np.random.shuffle(points)

# Split
split = int(0.8 * len(points))
train_data = points[:split]
test_data = points[split:]

# Train model
w, b, losses = train(train_data)
plot_loss_curve(losses)


# Evaluate
test_loss = loss_function(w, b, test_data)
print("Final Test Loss:", test_loss)

# Prepare the data
x_test = test_data[:, :-1]
y_test = test_data[:, -1]

# Make prediction
y_pred = np.dot(x_test, w) + b

# Plot evaluation
plot_actual_vs_predicted(y_test, y_pred)
plot_residuals(y_test, y_pred)
