from src.data_preprocessing import create_data_generators  # Adjust the import based on your structure
from src.model import build_resnet_model  # Adjust the import based on your structure
import os




# Change working directory to the root of the project
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Define directories
train_dir = 'data/train'
val_dir = 'data/val'
test_dir = 'data/test'

def evaluate_model():
    # Step 1: Create data generators
    train_generator, val_generator, test_generator = create_data_generators(train_dir, val_dir, test_dir)

    # Step 2: Build the model
    input_shape = (150, 150, 3)  # Adjust based on your input size
    model = build_resnet_model(input_shape)

    # Step 3: Train the model
    model.fit(train_generator, validation_data=val_generator, epochs=10)

    # Step 4: Save the trained model
    model.save('models/trained_resnet_model.h5')

