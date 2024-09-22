# test_preprocess_data.py
import pytest
from src.data_preprocessing import create_data_generators
import os


@pytest.fixture
def setup_directories(tmpdir):
    """Create temporary directories and image files for testing"""
    # Create temporary train, val, and test directories
    train_dir = tmpdir.mkdir("train")
    val_dir = tmpdir.mkdir("val")
    test_dir = tmpdir.mkdir("test")

    # Create class subdirectories in each
    for dir in [train_dir, val_dir, test_dir]:
        dir.mkdir("NORMAL")
        dir.mkdir("PNEUMONIA")

    # Add some dummy image files to the directories (e.g., 3 images per class)
    for subdir in ['NORMAL', 'PNEUMONIA']:
        for i in range(3):
            # Create a dummy image file
            image_file = train_dir.join(subdir, f"image_{i}.jpg")
            image_file.write("dummy image data")

            image_file_val = val_dir.join(subdir, f"image_{i}.jpg")
            image_file_val.write("dummy image data")

            image_file_test = test_dir.join(subdir, f"image_{i}.jpg")
            image_file_test.write("dummy image data")

    return str(train_dir), str(val_dir), str(test_dir)


def test_data_generators_creation(setup_directories):
    """Test if the data generators are correctly created"""
    train_dir, val_dir, test_dir = setup_directories

    # Call the function to create data generators
    train_generator, val_generator, test_generator = create_data_generators(train_dir, val_dir, test_dir)

    # Check if the generators were created and have the right number of images
    assert train_generator.samples == 6  # 3 NORMAL + 3 PNEUMONIA
    assert val_generator.samples == 6  # 3 NORMAL + 3 PNEUMONIA
    assert test_generator.samples == 6  # 3 NORMAL + 3 PNEUMONIA

    # Check that the class indices were correctly identified
    assert set(train_generator.class_indices.keys()) == {"NORMAL", "PNEUMONIA"}


def test_image_size_and_class_mode(setup_directories):
    """Test if the images are resized correctly and class mode is binary"""
    train_dir, val_dir, test_dir = setup_directories

    # Call the function to create data generators
    train_generator, val_generator, test_generator = create_data_generators(train_dir, val_dir, test_dir)

    # Get a batch of images and check their size
    images, labels = next(train_generator)
    assert images.shape == (32, 150, 150, 3)  # 32 images in a batch, 150x150, 3 channels (RGB)

    # Check if the class mode is binary (labels are 0 or 1)
    assert labels.ndim == 1  # For binary classification
    assert set(labels) <= {0, 1}  # Labels should only be 0 or 1
