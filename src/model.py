# model.py
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def build_resnet_model(input_shape, num_classes=1):
    """
    Build and fine-tune ResNet50 for image classification.
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes (default: 1 for binary classification)
    Returns:
        model: Fine-tuned ResNet50 model
    """
    # Load ResNet50 with pre-trained weights, excluding the top fully-connected layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base model layers to prevent them from being updated during training
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Global average pooling to reduce dimensionality
    x = Dense(128, activation='relu')(x)  # Add fully connected layer
    x = Dense(64, activation='relu')(x)  # Another fully connected layer

    # Output layer for binary classification
    predictions = Dense(num_classes, activation='sigmoid')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model (use binary crossentropy for binary classification)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
