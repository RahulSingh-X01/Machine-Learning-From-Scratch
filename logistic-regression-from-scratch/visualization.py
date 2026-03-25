import matplotlib.pyplot as plt

def plot_loss_curve(losses):
    plt.figure()
    plt.plot(losses)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def plot_probability_distribution(y_true, y_pred):
    import matplotlib.pyplot as plt

    plt.figure()

    # Separate classes
    plt.hist(y_pred[y_true == 0], bins=20, alpha=0.5, label="Class 0")
    plt.hist(y_pred[y_true == 1], bins=20, alpha=0.5, label="Class 1")

    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Probability Distribution")
    plt.legend()
    plt.show()
    
def plot_confusion_matrix(y_true, y_pred):
    import numpy as np
    import matplotlib.pyplot as plt

    # Convert probability → class
    y_pred_class = (y_pred >= 0.5).astype(int)

    TP = np.sum((y_true == 1) & (y_pred_class == 1))
    TN = np.sum((y_true == 0) & (y_pred_class == 0))
    FP = np.sum((y_true == 0) & (y_pred_class == 1))
    FN = np.sum((y_true == 1) & (y_pred_class == 0))

    cm = np.array([[TN, FP],
                   [FN, TP]])

    plt.figure()
    plt.imshow(cm)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()