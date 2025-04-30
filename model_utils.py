import tensorflow as tf
from tensorflow.keras import layers, models, applications

def create_model(architecture='Custom CNN', input_shape=(32, 32, 3), num_classes=10, learning_rate=0.001):
    """
    Create a CNN model for image classification.
    
    Args:
        architecture: Type of architecture to use ('Custom CNN', 'MobileNetV2', 'ResNet50', 'EfficientNetB0')
        input_shape: Shape of the input images
        num_classes: Number of classes for classification
        learning_rate: Learning rate for the optimizer
        
    Returns:
        Compiled TensorFlow model
    """
    if architecture == 'Custom CNN':
        model = create_custom_cnn(input_shape, num_classes)
    elif architecture == 'MobileNetV2':
        model = create_mobilenetv2(input_shape, num_classes)
    elif architecture == 'ResNet50':
        model = create_resnet50(input_shape, num_classes)
    elif architecture == 'EfficientNetB0':
        model = create_efficientnet(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_custom_cnn(input_shape, num_classes):
    """Create a custom CNN model."""
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_mobilenetv2(input_shape, num_classes):
    """Create a MobileNetV2-based model."""
    # Create a base pre-trained model
    base_model = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None  # We'll train from scratch due to input size differences
    )
    
    # Freeze the base model
    base_model.trainable = True
    
    # Create new model on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_resnet50(input_shape, num_classes):
    """Create a ResNet50-based model."""
    # Create a base pre-trained model
    base_model = applications.ResNet50V2(
        input_shape=input_shape,
        include_top=False,
        weights=None  # We'll train from scratch due to input size differences
    )
    
    # Freeze the base model
    base_model.trainable = True
    
    # Create new model on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_efficientnet(input_shape, num_classes):
    """Create an EfficientNetB0-based model."""
    # Create a base pre-trained model
    base_model = applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights=None  # We'll train from scratch due to input size differences
    )
    
    # Freeze the base model
    base_model.trainable = True
    
    # Create new model on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
