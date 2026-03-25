import numpy as np
import pandas as pd
from logistic_regression_model import train, loss_function
from visualization import plot_loss_curve, plot_probability_distribution, plot_confusion_matrix

#--------Load the dataset--------#
data = pd.read_csv(r"Machine-Learning-From-Scratch\logistic-regression-from-scratch\dataset.csv")
points = data.values.copy()

#--------Sufflle the dataset-----#
np.random.seed(42)
np.random.shuffle(points)

#---------Train-test split-------#
split = int(0.8*len(points))
train_data = points[:split]
test_data = points[split:]

#---------Model training---------#
w, b, losses = train(train_data)
plot_loss_curve(losses)

#---------Evaluation-------------#
test_loss = loss_function(w, b, test_data)
print("Final test loss : ", test_loss)

x_test = test_data[:, :-1]
y_test = test_data[:, -1]

# Make prediction
y_pred = 1/(1+np.exp(-(np.dot(x_test, w)+b)))

#---------Visualization-------------#

# Probability distribution
plot_probability_distribution(y_test, y_pred)

# Confusion matrix
plot_confusion_matrix(y_test, y_pred)





