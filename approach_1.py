import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
path = 'mnist.npz'


class Autoencoder:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        input_layer = Input(shape=(28, 28, 1))
        zero_pad = ZeroPadding2D((2, 2))(input_layer)
        conv1 = Conv2D(112, (5, 5), activation='elu', padding='same')(zero_pad)
        #pool1 = MaxPooling2D((2, 2), padding='same')(conv1)
        # conv2 = Conv2D(112, (5, 5), activation='elu', padding='same')(pool1)
        # pool2 = MaxPooling2D((2, 2), padding='same')(conv2)
        # conv3 = Conv2D(32, (5, 5), activation='elu', padding='same')(pool2)
        # pool3 = MaxPooling2D((2, 2), padding='same')(conv3)
        # conv4 = Conv2D(8, (5, 5), activation='elu', padding='same')(pool3)
        encoder = MaxPooling2D((2, 2), padding='same')(conv1)

        conv_1 = Conv2D(112, (5, 5), activation='elu', padding='same')(encoder)
        upsample1 = UpSampling2D((2, 2))(conv_1)
        # conv_2 = Conv2D(8, (5, 5), activation='elu', padding='same')(upsample1)
        # upsample2 = UpSampling2D((2, 2))(conv_2)
        # conv_3 = Conv2D(16, (5, 5), activation='elu', padding='same')(upsample2)
        # upsample3 = UpSampling2D((2, 2))(conv_3)
        conv_4 = Conv2D(1, (3, 3), activation='elu', padding='same')(upsample1)
        #upsample4 = UpSampling2D((2, 2))(conv_4)
        decoder = Cropping2D((2, 2))(conv_4)

        autoencoder = Model(input_layer, decoder)
        optimizer = keras.optimizers.Adam(learning_rate=0.0029044)
        autoencoder.compile(optimizer=optimizer, loss='mse')
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
    autoencoder.model.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test), callbacks=callbacks)
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
    DataHandler.visualize_data(x_train)
    
    autoencoder = train_autoencoder(x_train, x_test)
    decoded_imgs = autoencoder.model.predict(x_test)
    
    visualize_results(x_test, decoded_imgs)
