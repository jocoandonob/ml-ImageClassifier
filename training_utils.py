import tensorflow as tf
import numpy as np

def train_model(model, train_dataset, valid_dataset, epochs=10, batch_size=32, augmentation_options=None):
    """
    Train the model on the provided dataset.
    
    Args:
        model: TensorFlow model to train
        train_dataset: Training dataset
        valid_dataset: Validation dataset
        epochs: Number of epochs to train for
        batch_size: Batch size for training
        augmentation_options: Dictionary of data augmentation options
        
    Returns:
        Training history
    """
    # Optional callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history

def evaluate_model(model, test_dataset):
    """
    Evaluate the model on the test dataset.
    
    Args:
        model: Trained TensorFlow model
        test_dataset: Test dataset
        
    Returns:
        test_loss, test_accuracy, predicted_classes, true_classes
    """
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_dataset)
    
    # Get predictions
    y_pred = []
    y_true = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)
        
        y_pred.extend(pred_classes)
        y_true.extend(true_classes)
    
    return test_loss, test_accuracy, np.array(y_pred), np.array(y_true)
