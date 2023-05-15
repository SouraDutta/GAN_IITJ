from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, LeakyReLU, Input, Lambda
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import optimizers, regularizers
import numpy as np

from sklearn import metrics
from keras.utils import to_categorical

from sklearn.cluster import KMeans
import joblib
from scipy.io import loadmat

class MyEarlyStopping(EarlyStopping):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.baseline_attained = False

    def on_epoch_end(self, epoch, logs=None):
        if not self.baseline_attained:
            current = self.get_monitor_value(logs)
            if current is None:
                return

            if self.monitor_op(current, self.baseline):
                if self.verbose > 0:
                    print('Baseline attained.')
                self.baseline_attained = True
                self.model.stop_training = True
            else:
                return

        super(MyEarlyStopping, self).on_epoch_end(epoch, logs)


img_rows = 28
img_cols = 20

########################## Load frey ##########################

file = loadmat("./frey_rawface.mat", squeeze_me=True, struct_as_record=False)
file = file["ff"].T.reshape((-1, img_rows, img_cols))

np.random.seed(42)
original_dim = 28 * 20
x_train = file[:1800]
x_test = file[1800:1900]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), original_dim))
x_test = x_test.reshape((len(x_test), original_dim))

print(x_train.shape)
print(x_test.shape)


intermediate_dim = 20 #embedding to 20
latent_dim = 2

inputs = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)

from keras import backend as K

########################## Sampling function ##########################

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = Lambda(sampling)([z_mean, z_log_sigma])


########################## create encoder ##########################
encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

########################## create decoder ##########################
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)
decoder = Model(latent_inputs, outputs, name='decoder')

########################## create VAE ##########################

outputs = decoder(encoder(inputs)[2])
var_ae = Model(inputs, outputs, name='vae_mlp')

reconstruction_loss = binary_crossentropy(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
var_ae_loss = K.mean(reconstruction_loss + kl_loss)
var_ae.add_loss(var_ae_loss)
var_ae.compile(optimizer='adam')

print(var_ae.summary())
batch_size = 32



########################## Training ##########################
es = MyEarlyStopping(monitor='val_acc', mode='max', baseline=0.65, patience = 0)
var_ae.fit(
        x_train, x_train,
        epochs=100,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test, x_test)
        )

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# print(encoded_imgs)
# print(decoded_imgs)

var_ae.save_weights('vae_model.h5')

########################## Display faces ##########################

import matplotlib.pyplot as plt

n = 15  # figure with 15x15 faces
digit_size = 28
figure = np.zeros((digit_size * n, 20 * n))
# sampling within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, 20)
        figure[i * digit_size: (i + 1) * digit_size,
               j * 20: (j + 1) * 20] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()