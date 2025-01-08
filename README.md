# Cloud Cover Nowcasting
This project aims to predict cloud cover using a sequence-to-sequence ConvLSTM (Convolutional Long Short-Term Memory) model. The goal is to predict future cloud cover based on past satellite images. The model uses satellite .tif images taken at regular intervals to forecast cloud patterns, aiding in weather prediction and climate monitoring.

## Features
Satellite Image Processing: The model processes `.tif` images, which are commonly used in satellite data, providing high-resolution, geospatial information.
ConvLSTM for Sequential Prediction: The ConvLSTM model is used to learn and predict the temporal dynamics of cloud cover. ConvLSTM combines the power of convolutional layers (for spatial feature extraction) with `LSTM (Long Short-Term Memory)` layers (for sequence learning), making it ideal for spatiotemporal data like satellite imagery.
Model Evaluation: The model's performance is evaluated using metrics like Structural Similarity Index (SSIM) and Mean Squared Error (MSE). SSIM helps in evaluating the similarity between the predicted and actual cloud cover images, while MSE gives a quantitative measure of prediction error.
Project Overview
## What is ConvLSTM?
ConvLSTM is an advanced deep learning architecture that is particularly suitable for processing spatiotemporal data. Unlike regular LSTMs, which operate on sequences of scalar values, ConvLSTMs apply convolution operations within the LSTM structure, allowing the model to capture both spatial and temporal dependencies in the data.

Convolutional Layers: These layers capture spatial patterns in images, such as cloud structures, edges, and textures.
LSTM Layers: These layers allow the model to learn temporal dependencies, meaning the model can learn how cloud patterns evolve over time.
Sequence-to-Sequence Learning: The model is trained to predict the next frames in a sequence based on the previous ones. This is particularly useful for tasks like weather forecasting, where the past data influences future predictions.
Time Series Forecasting with ConvLSTM
The dataset consists of a series of satellite images (each representing cloud cover at different time intervals). Time series forecasting using ConvLSTM allows the model to predict the next frames in the sequence, making it an ideal tool for nowcasting cloud cover. Each sequence of images provides context to help the model make accurate predictions about future cloud cover, taking into account both spatial and temporal factors.

## Dataset
The dataset consists of .tif images (grayscale cloud cover images) that are loaded into memory and preprocessed for model training. The images are organized into sequences of frames to enable the model to learn the temporal evolution of cloud cover.



## Features in short
- Processes satellite `.tif` images.
- Uses ConvLSTM for sequential prediction.
- Evaluates model accuracy using SSIM and MSE.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Add `.tif` images to the `data/` directory.

## Run Training
The model uses ConvLSTM layers to process the sequences of images. It learns spatial features from each frame and temporal patterns from the sequence, enabling it to predict future cloud cover images. The model is trained using mean absolute error (MAE) as the loss function, and it is optimized using the Adam optimizer.
`python /train.py`

## Run Evaluate
Once the model is trained, you can evaluate its performance on a validation set using the following command:
`python /evaluate.py`

## Run Predict
To make predictions on a new set of satellite images, use the following command:
`python /predict.py`
