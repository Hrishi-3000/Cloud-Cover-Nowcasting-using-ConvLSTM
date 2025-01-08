import tensorflow as tf
from tensorflow.keras.layers import (
    ConvLSTM2D, Input, Conv2D, BatchNormalization, Add, ReLU, TimeDistributed
)

def build_residual_convlstm_model_seq2seq(input_shape):
    input_layer = Input(shape=input_shape)
    
    # First ConvLSTM layer with residual connection
    x = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True)(input_layer)
    x = BatchNormalization()(x)
    res = x  # Save the residual

    # Second ConvLSTM layer
    x = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)

    # Residual connection
    x = Add()([x, res])

    # Third ConvLSTM layer with residual connection, returning the entire sequence
    x = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)

    # Apply Conv2D and ReLU to each frame in the sequence using TimeDistributed
    x = TimeDistributed(Conv2D(128, (3, 3), padding='same'))(x)
    x = TimeDistributed(ReLU())(x)

    # Final Conv2D layer to predict the sequence of frames
    output_layer = TimeDistributed(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model
