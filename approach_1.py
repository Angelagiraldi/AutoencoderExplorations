import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Activation, Cropping2D, Dense
from tensorflow.keras.models import Model


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau



path = 'mnist.npz'
# Loading our data
with np.load(path, allow_pickle=True) as f:  
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']



# Understanding the shape of the dataset
print(x_train[0].shape)

# Visualizing the data
plt.imshow(x_train[2], cmap="gray")
plt.savefig('visualization_traindata.pdf')


# Normalizing the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# Creating the input layer with the full shape of the image
input_layer = Input(shape=(28, 28, 1))

# Note: If utilizing a deep neural network without convolution, ensure that the dimensions are multiplied and converted 
# accordingly before passing through further layers of the encoder architecture.

zero_pad = ZeroPadding2D((2, 2))(input_layer)

# First layer
conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(zero_pad)
pool1 = MaxPooling2D((2, 2), padding='same')(conv1)

# Second layer
conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D((2, 2), padding='same')(conv2)

# Third layer
conv3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2)
pool3 = MaxPooling2D((2, 2), padding='same')(conv3)

# Final layer
conv4 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool3)

# Encoder architecture
encoder = MaxPooling2D((2, 2), padding='same')(conv4)


# First reconstructing decoder layer
conv_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
upsample1 = UpSampling2D((2, 2))(conv_1)

# Second reconstructing decoder layer
conv_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(upsample1)
upsample2 = UpSampling2D((2, 2))(conv_2)

# Third decoder layer
conv_3 = Conv2D(16, (3, 3), activation='relu', padding='same')(upsample2)
upsample3 = UpSampling2D((2, 2))(conv_3)

# First reconstructing decoder layer
conv_4 = Conv2D(1, (3, 3), activation='relu', padding='same')(upsample3)
upsample4 = UpSampling2D((2, 2))(conv_4)

# Decoder architecture
decoder = Cropping2D((2, 2))(upsample4)

# Creating and compiling the model

autoencoder = Model(input_layer, decoder)
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())

# Creating Callbacks
tensorboad_results = TensorBoard(log_dir='autoencoder_logs_fashion/')
checkpoint = ModelCheckpoint("best_model_fashion.h5", monitor="val_loss", save_best_only=True)
early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=False)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=4, min_lr=0.000001)

# Training our autoencoder model
autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[checkpoint, early_stop, tensorboad_results, reduce_lr])

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig('visualization_results.pdf')