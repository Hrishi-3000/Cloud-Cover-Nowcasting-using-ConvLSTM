import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, structural_similarity as ssim

def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test.flatten(), predictions.flatten())
    ssim_score = ssim(y_test[0, -1, :, :, 0], predictions[0, -1, :, :, 0], data_range=1.0)
    print(f"Mean Squared Error: {mse}")
    print(f"Structural Similarity Index: {ssim_score}")

    return predictions

def visualize_predictions(x_test, y_test, predictions, idx=0):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(x_test[idx, -1, :, :, 0], cmap='gray')
    axes[0].set_title("Input Frame")

    axes[1].imshow(y_test[idx, -1, :, :, 0], cmap='gray')
    axes[1].set_title("True Frame")

    axes[2].imshow(predictions[idx, -1, :, :, 0], cmap='gray')
    axes[2].set_title("Predicted Frame")

    plt.show()
