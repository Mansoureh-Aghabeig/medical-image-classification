import pytest
from keras.models import Model
from keras.layers import Input
from src.model import build_resnet_model  # Adjust the import based on your project structure

def test_model_output_shape():
    # Test if the model outputs the correct shape for binary classification
    input_shape = (150, 150, 3)  # Example input shape
    model = build_resnet_model(input_shape)

    # Get the model summary to verify the output layer
    model_summary = model.summary()
    assert model.output_shape == (None, 1)  # Output should be (batch_size, 1) for binary classification

def test_model_compilation():
    # Test if the model compiles without errors
    input_shape = (150, 150, 3)
    model = build_resnet_model(input_shape)

    try:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    except Exception as e:
        pytest.fail(f"Model compilation failed: {e}")

def test_model_structure():
    # Test if the model structure contains expected layers
    input_shape = (150, 150, 3)
    model = build_resnet_model(input_shape)

    layer_names = [layer.name for layer in model.layers]
    assert 'global_average_pooling2d' in layer_names  # Check for global average pooling layer
    assert 'dense' in layer_names  # Check for dense layers
    assert 'dense_1' in layer_names  # Check for the output layer

def test_model_prediction_shape():
    # Test if the model can make a prediction and check output shape
    input_shape = (150, 150, 3)
    model = build_resnet_model(input_shape)

    import numpy as np
    dummy_input = np.random.rand(1, 150, 150, 3)  # Create a dummy input with the correct shape
    predictions = model.predict(dummy_input)

    assert predictions.shape == (1, 1)  # Output shape should be (1, 1) for binary classification
