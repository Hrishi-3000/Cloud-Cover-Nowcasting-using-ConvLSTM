import os
import cv2
import numpy as np

def load_images_from_folder(folder_path, img_size=(200, 200)):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.tif'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = img.astype('float32') / np.max(img)  # Normalize
                images.append(img)
            else:
                print(f"Warning: {img_path} could not be read.")
    print(f"Total images loaded from {folder_path}: {len(images)}")
    return np.array(images)

def create_sequences(data, sequence_length):
    num_sequences = data.shape[0] - sequence_length + 1
    return np.array([data[i:i + sequence_length] for i in range(num_sequences)])

def create_shifted_frames(data):
    x = data[:, :-1, :, :, :]  # Frames 0 to n-1
    y = data[:, 1:, :, :, :]  # Frames 1 to n
    return x, y
