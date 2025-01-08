import numpy as np
from tensorflow.keras.models import load_model
from data_loader import load_images_from_folder, create_sequences

def predict_next_frame(model_path, input_sequence):
    model = load_model(model_path)
    predictions = model.predict(input_sequence)
    return predictions

if __name__ == "__main__":
    folder_path = "/path/to/new/data"
    img_size = (200, 200)
    sequence_length = 5

    # Load and preprocess data
    dataset = load_images_from_folder(folder_path, img_size=img_size)
    dataset = np.expand_dims(dataset, axis=-1)
    sequences = create_sequences(dataset, sequence_length)

    # Load trained model and predict
    model_path = "best_model.keras"
    predictions = predict_next_frame(model_path, sequences)
    print("Predictions generated for the input sequence.")
