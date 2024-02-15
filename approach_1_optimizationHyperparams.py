import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras import backend as K
from keras_tuner import Hyperband


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
path = 'mnist.npz'


def build_autoencoder(hp):
    input_img = Input(shape=(28, 28, 1))
    x = ZeroPadding2D((2, 2))(input_img)

    activation = hp.Choice('activation', values=['relu', 'elu', 'sigmoid', 'tanh'])
    kernel_size = hp.Choice('kernel_size', values=[3, 5])
    for i in range(hp.Int('num_conv_layers', 1, 5)):
        filters = hp.Int(f'conv_{i}_filters', min_value=16, max_value=128, step=16)
        x = Conv2D(filters, (kernel_size, kernel_size), activation=activation, padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
    
    for i in reversed(range(hp.Int('num_conv_layers', 1, 5))):
        filters = hp.Int(f'conv_{i}_filters', min_value=16, max_value=128, step=16)
        x = Conv2D(filters, (kernel_size, kernel_size), activation=activation, padding='same')(x)
        x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    x = Cropping2D((2, 2))(x)
    autoencoder = Model(input_img, x)
    optimizer = keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'))
    loss_function = hp.Choice('loss_function', values=['mean_squared_error', 'binary_crossentropy', 'mean_absolute_error'])
    autoencoder.compile(optimizer=optimizer, loss=loss_function)
    
    return autoencoder

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

    tuner = Hyperband(
        build_autoencoder,
        objective='val_loss',
        max_epochs=10,
        directory='hyperband',
        project_name='mnist_autoencoder',
        overwrite=True,
        seed=42
    )

    tuner.search(x_train, x_train, epochs=10, validation_split=0.2, callbacks=[EarlyStopping('val_loss', patience=5)])

    best_model = tuner.get_best_models(num_models=1)[0]
    print("Summary of the best model found:")
    best_model.summary()
    
