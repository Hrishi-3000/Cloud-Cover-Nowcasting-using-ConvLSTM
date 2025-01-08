# Cloud Cover Nowcasting

This project predicts cloud cover using a sequence-to-sequence ConvLSTM model.

## Features
- Processes satellite `.tif` images.
- Uses ConvLSTM for sequential prediction.
- Evaluates model accuracy using SSIM and MSE.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Add `.tif` images to the `data/` directory.

## Run Training

`python /train.py`

## Run Evaluate

`python /evaluate.py`

## Run Predict

`python /predict.py`
