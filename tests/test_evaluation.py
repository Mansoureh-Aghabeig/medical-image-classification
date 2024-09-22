import pytest
from src.evaluation import evaluate_model  # Adjust the import based on your structure

# Mock the necessary components
from unittest.mock import patch

@pytest.fixture
def mock_data_generators():
    """Mock data generators for testing."""
    class MockGenerator:
        def __iter__(self):
            return iter([(None, None)] * 10)  # Mock 10 batches of data

    return MockGenerator()

def test_evaluate_model(mock_data_generators):
    with patch('src.data_preprocessing') as mock_create_data_generators, \
         patch('src.model') as mock_build_model:

        # Mocking the create_data_generators function
        mock_create_data_generators.return_value = (mock_data_generators, mock_data_generators, mock_data_generators)

        # Mocking the model
        mock_model = patch('tensorflow.keras.Model').start()
        mock_model.fit.return_value = None  # Mock fit to do nothing
        mock_model.evaluate.return_value = (0.1, 0.9)  # Mock evaluation to return loss and accuracy
        mock_build_model.return_value = mock_model

        # Call the evaluate_model function
        evaluate_model()

        # Assertions
        mock_create_data_generators.assert_called_once()  # Ensure data generators were created
        mock_build_model.assert_called_once()  # Ensure the model was built
        mock_model.fit.assert_called_once()  # Ensure the model was trained
        assert mock_model.evaluate.return_value == (0.1, 0.9)  # Check the mocked evaluation output

        print("All tests passed!")

if __name__ == "__main__":
    pytest.main()
