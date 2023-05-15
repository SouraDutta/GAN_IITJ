from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, LeakyReLU, Input
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras import optimizers, regularizers
import numpy as np

from sklearn import metrics
from keras.utils import to_categorical

from sklearn.cluster import KMeans
import joblib

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

(x_train,y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

encoding_dim = 32  # 32 floats 
input_img = Input(shape=(784,))
encoded_img = Input(shape=(32,))

########################## Create Sparse Encoder ##########################
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)

########################## Create Decoder ##########################
decoded = Dense(784, activation='sigmoid')(encoded_img)

# This model maps an input to its reconstruction
encoder = Model(inputs=input_img, outputs=encoded)
decoder = Model(inputs=encoded_img, outputs=decoded)
autoencoder = Model(inputs=input_img, outputs=decoder(encoder(input_img)))


########################## Create autoencoder ##########################

autoencoder.compile(optimizer=optimizers.Adam(lr=1e-4), loss='binary_crossentropy')


print(autoencoder.summary())
batch_size = 100



########################## Training ##########################
es = MyEarlyStopping(monitor='val_acc', mode='max', baseline=0.65, patience = 0)
autoencoder.fit(
        x_train, x_train,
        epochs=50,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test, x_test)
        )

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# print(encoded_imgs)
# print(decoded_imgs)

autoencoder.save_weights('sae_model.h5')

########################## Display digits ##########################

import matplotlib.pyplot as plt

n = 10  
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

########################## KMeans ##########################

Kmean = KMeans(n_clusters=10, init='k-means++', max_iter=4000, n_init=100, verbose=1)
Kmean.fit(encoded_imgs)

order_centroids = Kmean.cluster_centers_.argsort()[:, ::-1]
terms = y_test

for i in range(10):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

