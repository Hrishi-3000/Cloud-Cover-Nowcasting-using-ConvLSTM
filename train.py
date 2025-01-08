import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_loader import load_images_from_folder, create_sequences, create_shifted_frames
from model_builder import build_residual_convlstm_model_seq2seq

# Parameters
folder_path = "/home/hrishikesh2003/Data/DataNewApproch"
image_size = (200, 200)
sequence_length = 5
batch_size = 4
epochs = 20

# Load and preprocess data
dataset = load_images_from_folder(folder_path, img_size=image_size)
dataset = np.expand_dims(dataset, axis=-1)
sequences = create_sequences(dataset, sequence_length)

# Split data
train_sequences, val_sequences = train_test_split(sequences, test_size=0.1, shuffle=False)
x_train, y_train = create_shifted_frames(train_sequences)
x_val, y_val = create_shifted_frames(val_sequences)

# Build model
model = build_residual_convlstm_model_seq2seq(input_shape=(sequence_length - 1, *image_size, 1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')

# Callbacks
checkpoint = ModelCheckpoint(
    'best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1
)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

# Train model
if x_train.shape[0] > 0 and y_train.shape[0] > 0:
    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[checkpoint, early_stopping]
    )
else:
    print("Not enough training data to train the model.")
