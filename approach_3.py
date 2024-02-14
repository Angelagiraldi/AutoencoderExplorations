import os

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Activation, Cropping2D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
path = 'mnist.npz'

class Autoencoder:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        encoding_dim = 32  
        # Input Sample
        input_img = Input(shape=(784,))
        # encoder network
        encoder = Dense(encoding_dim, activation='relu')(input_img)
        # decoder network
        decoder = Dense(784, activation='sigmoid')(encoder)


        autoencoder = Model(input_img, decoder)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def summary(self):
        return self.model.summary()

class DataHandler:
    @staticmethod
    def load_and_normalize_data():
        
        with np.load(path, allow_pickle=True) as f:  
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        return x_train, x_test

    @staticmethod
    def visualize_data(data, title='Train Data Visualization', save_as='visualization_traindata.pdf'):
        plt.imshow(data[2], cmap="gray")
        plt.title(title)
        plt.savefig(save_as)
        plt.show()

def train_autoencoder(x_train, x_test):
    callbacks = [
        ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=8),
        TensorBoard(log_dir='./autoencoder_logs/'),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=4, min_lr=0.000001)
    ]

    autoencoder = Autoencoder()
    autoencoder.model.fit(x_train, x_train, epochs=100, batch_size=128, shuffle=True, validation_data=(x_test, x_test), callbacks=callbacks)
    return autoencoder

def visualize_results(x_test, decoded_imgs, n=10, save_as='visualization_results.pdf'):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

        # Display reconstruction
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

    plt.savefig(save_as)
    plt.show()

# Main execution
if __name__ == "__main__":
    x_train, x_test = DataHandler.load_and_normalize_data()
    
    autoencoder = train_autoencoder(x_train, x_test)
    decoded_imgs = autoencoder.model.predict(x_test)
    
    visualize_results(x_test, decoded_imgs)


