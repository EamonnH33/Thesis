import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir


fr = pd.read_csv("FR_cleaned_data.csv")

### Remove problematic lots

remove_list = []
fr = fr[fr["index"].isin(remove_list) == False]

# reset index

Y_train, Y_validation, Y_test = [], [], []
X_train, X_validation, X_test = [], [], []
lot = []
test_lot = []

train_index = [i for i in range(101)]
validation_index = [i for i in range(101,118)]
test_index = [i for i in range(118,135)]

# get the path/directory
folder_dir = "./FR_images_3/"
for images in os.listdir(folder_dir):

    # check if the image ends with png
    if (images.endswith(".jpg")):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ### get y ###

        # get lot index
        d_lot = int(images[4:images.find("_img")])
        lot.append(d_lot)
        # get weight
        d_weight = int(fr[fr["index"] == d_lot].reset_index(drop=True).cat_weightKgs[0])
        # append to y

        if len(set(train_index).intersection(set([d_lot]))) == 1:

            Y_train += [d_weight]

        elif len(set(test_index).intersection(set([d_lot]))) == 1:

            Y_test += [d_weight]
            test_lot.append(d_lot)

        else:

            Y_validation += [d_weight]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ### get x ###

        # load image
        img = image.load_img(folder_dir + images, target_size=(240, 427, 3))
        # transform to array
        img_array = image.img_to_array(img)
        # append to x

        if set(train_index).intersection(set([d_lot])):

            X_train += [img_array]

        elif set(test_index).intersection(set([d_lot])):

            X_test += [img_array]

        else:

            X_validation += [img_array]


Y_train, Y_validation, Y_test = np.array(Y_train), np.array(Y_validation), np.array(Y_test)
X_train, X_validation, X_test = np.array(X_train)[:, :, :, :3], np.array(X_validation)[:, :, :, :3], np.array(X_test)[:, :, :, :3]

print("Y dimensions", Y_train.shape, Y_validation.shape, Y_test.shape)
print("X dimensions", X_train.shape, X_validation.shape, X_test.shape)



### Define CNN

input_shape = (240, 427, 3)

model = Sequential()
model.add(Conv2D(filters = 64, kernel_size = (12, 12), activation='relu',
                 input_shape = input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (12, 12), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(8, 8)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 128, kernel_size = (12, 12), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 128, kernel_size = (12, 12), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(8, 8)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
#model.add(Dense(n_classes, activation='softmax'))
model.add(Dense(1, activation='relu'))
learning_rate = 0.0001
model.compile(loss = 'mae',
              optimizer = Adam(learning_rate))
model.summary()

### Fit Model

history = model.fit( X_train, Y_train,
                    epochs = 10, batch_size = 100, verbose=1,
                   validation_data = (X_validation, Y_validation))


plt.figure(figsize=(6, 5))
# training loss
plt.plot(history.history['loss'], color='r')
#validation loss
plt.plot(history.history['val_loss'], color='g')
print(plt.show())


### Test Lot

print(test_lot)

