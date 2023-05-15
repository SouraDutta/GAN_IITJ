from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Reshape
from keras.layers import Activation, Dropout, Flatten, Dense, LeakyReLU, Input, Lambda, BatchNormalization
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
x_train = file
x_train = x_train.astype('float32') / 255.

# x_train = x_train.reshape((len(x_train), original_dim))
# x_test = x_test.reshape((len(x_test), original_dim))
x_train = x_train.reshape(-1, img_rows, img_cols, 1).astype(np.float32)


print(x_train.shape)

########################## Create Discriminator ##########################

discriminator = Sequential()
depth = 64
dropout = 0.4
# In: 28 x 28 x 1, depth = 1
# Out: 14 x 14 x 1, depth=64
input_shape = (img_rows, img_cols, 1)
discriminator.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,\
padding='same', activation=LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(dropout))
discriminator.add(Conv2D(depth*2, 5, strides=2, padding='same',\
activation=LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(dropout))
discriminator.add(Conv2D(depth*4, 5, strides=2, padding='same',\
activation=LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(dropout))
discriminator.add(Conv2D(depth*8, 5, strides=1, padding='same',\
activation=LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(dropout))
# Out: 1-dim probability
discriminator.add(Flatten())
discriminator.add(Dense(1))
discriminator.add(Activation('sigmoid'))
print(discriminator.summary())

discriminator.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8), metrics=['accuracy'])

########################## Create Generator ##########################

generator = Sequential()
dropout = 0.4
depth = 254
dim = 7

generator.add(Dense(7*5*depth, input_dim=100))
generator.add(BatchNormalization(momentum=0.9))
generator.add(Activation('relu'))
generator.add(Reshape((7, 5, depth)))
generator.add(Dropout(dropout))
generator.add(UpSampling2D())
generator.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(Activation('relu'))
generator.add(UpSampling2D())
generator.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(Activation('relu'))
generator.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(Activation('relu'))

generator.add(Conv2DTranspose(1, 5, padding='same'))
generator.add(Activation('sigmoid'))
print(generator.summary())

generator.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8),metrics=['accuracy'])

gan = Sequential()

gan.add(generator)
gan.add(discriminator)

gan.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8),metrics=['accuracy'])

print(gan.summary())

train_steps=2000
batch_size=256

import matplotlib.pyplot as plt

########################## Train GAN and save faces ##########################

for i in range(train_steps):
    print(x_train.shape)
    images_train = x_train[np.random.randint(0,x_train.shape[0], size=batch_size)]
    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
    images_fake = generator.predict(noise)
    print(images_train.shape, images_fake.shape)
    x = np.concatenate((images_train, images_fake))
    y = np.ones([2*batch_size, 1])
    y[batch_size:, :] = 0
    d_loss = discriminator.train_on_batch(x, y)
    y = np.ones([batch_size, 1])
    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
    a_loss = gan.train_on_batch(noise, y)
    print("%d [discriminator loss: %f, acc: %f] [gan loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1], a_loss[0], a_loss[1]))
    if(i%50 == 0):
        filename = "frey_%d.png" % i
        noise = np.random.uniform(-1.0, 1.0, size=[16, 100])
        images = generator.predict(noise)
        
        plt.figure(figsize=(28,20))
        for i in range(images.shape[0]):
            print(i, images.shape[0])
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [img_rows, img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(filename)
    
        plt.show()
        plt.close('all')

    gan.save_weights('gan_model.h5')