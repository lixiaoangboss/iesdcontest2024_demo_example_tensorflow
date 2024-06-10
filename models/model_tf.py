import tensorflow as tf
from tensorflow.keras import layers, models

def AFNet():
    def residual_block(inputs, filters, kernel_size, strides):
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        return tf.nn.relu(x + inputs)

    inputs = layers.Input(shape=(1250, 1, 1))  # Adjust the shape according to your data
    x = layers.Conv2D(filters=3, kernel_size=(6, 1), strides=(2, 1), padding='valid', activation='relu')(inputs)
    x = layers.Conv2D(filters=5, kernel_size=(5, 1), strides=(2, 1), padding='valid', activation='relu')(x)
    x = layers.Conv2D(filters=10, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu')(x)
    x = layers.Conv2D(filters=10, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu')(x)
    x = layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu')(x)

    # Adding residual blocks
    x = residual_block(x, 20, (3, 1), 1)
    x = residual_block(x, 20, (3, 1), 1)

    # Flatten the output for the LSTM layer
    x = layers.Reshape((-1, x.shape[-1]))(x)
    
    # Adding LSTM layer
    x = layers.LSTM(20, return_sequences=True)(x)

    # Adding attention mechanism
    attention = layers.Attention()([x, x])
    x = layers.GlobalAveragePooling1D()(attention)

    x = layers.Dropout(0.7)(x)
    x = layers.Dense(10, activation='relu')(x)
    outputs = layers.Dense(2)(x)

    model = models.Model(inputs, outputs)
    return model
