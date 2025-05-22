import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Function to create and save the confusion matrix
def create_confusion_matrix(cm, labels, filename='confusion_matrix.png'):
    """
    Generate a confusion matrix heatmap from the input matrix and save it as a PNG file.
    
    Parameters:
    cm (2D list or array): The 3x3 confusion matrix.
    labels (list): The list of class labels for the confusion matrix.
    filename (str): The filename to save the PNG image (default is 'confusion_matrix.png').
    """
    # Convert confusion matrix to a numpy array if it isn't already
    cm = np.array(cm)

    # Create a heatmap using seaborn
    plt.figure(figsize=(6, 5))
    sns.set(font_scale=1.2)
    
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                     xticklabels=labels, yticklabels=labels, linewidths=0.5, linecolor='black')
    
    # Labels and titles
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    
    # Save the figure to a PNG file
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

# Example: input your confusion matrix values here
cm_values = [
    [50, 5, 2],  # True class 0 (Predicted as 0, 1, 2)
    [3, 45, 8],  # True class 1 (Predicted as 0, 1, 2)
    [1, 6, 47]   # True class 2 (Predicted as 0, 1, 2)
]

# List of class labels
labels = ['Class 0', 'Class 1', 'Class 2']

# Call the function to create and save the confusion matrix image
create_confusion_matrix(cm_values, labels, 'confusion_matrix.png')
