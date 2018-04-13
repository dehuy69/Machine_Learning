import pandas as pd
import numpy as np
import os
from scipy import misc
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import rmsprop, SGD, Adam
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
folder_work = '~/Self_Study/cifar-10'
train_folder = './train'
train_csv = 'trainLabels.csv'
num_classes = 10
def load_data():
    #os.chdir(folder_work)
    im_names = []
    for i in range(50001):
        im_name = str(i) + '.png'
        im_names.append(im_name)
    im_names.remove('0.png')

    file = open(train_csv,'rb')
    label = pd.read_csv(file)
    file.close()
    y = np.vstack(label.label)
    x = []
    for im_name in im_names:
        im = misc.imread(train_folder + '/' + im_name)
        x.append(im)
    x = np.array(x).astype('float32')/255. 
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.1,
                                                        random_state=42)
    return x_train, y_train, x_test, y_test
x_train, y_train, x_test, y_test = load_data()

def convert_label(y):
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    return encoded_Y
    
y_train = convert_label(y_train)
y_test = convert_label(y_test)

def model1():
    model =Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='softmax'))
    return model

def model2():
    model =Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model
def model3():
    model =Sequential()
    model.add(Conv2D(32, (5, 5), padding='same',
                 input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model
# choose model
model = model1()
# choose optimizer
opt1 = rmsprop(lr=0.0001, decay=1e-6)
opt2 = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
opt3 = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt2,
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=(x_test, y_test)
          )
acc = model.evaluate(x_test, y_test)[1]
num_loop = 1
while acc < 0.9:
    print (num_loop)
    model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=(x_test, y_test)
          )
    acc = model.evaluate(x_test, y_test)[1]
    num_loop +=1
