import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_training_history(history):
    """
    Plot the training and validation accuracy and loss.
    
    Args:
        history: History object returned by model.fit()
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy subplot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss subplot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_sample_images(images, labels, class_names, num_rows=5, num_cols=5):
    """
    Plot a grid of sample images with their class labels.
    
    Args:
        images: Array of images
        labels: Array of class indices
        class_names: List of class names
        num_rows: Number of rows in the grid
        num_cols: Number of columns in the grid
        
    Returns:
        Matplotlib figure
    """
    num_images = min(num_rows * num_cols, len(images))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i])
            ax.set_title(class_names[labels[i]])
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot a confusion matrix.
    
    Args:
        y_true: Array of true class indices
        y_pred: Array of predicted class indices
        class_names: List of class names
        
    Returns:
        Matplotlib figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Normalized Confusion Matrix')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig
