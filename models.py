# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras import applications, optimizers, losses, activations, models
from keras.applications import InceptionResNetV2
from keras.callbacks import TensorBoard
from time import time
def inception():
    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Convolution2D(32,(3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Adding a second convolutional layer
    classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(output_dim = 128, activation = 'relu'))
    classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
   # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    return classifier

#model predicts only 2 outcomes and is set for transfer learning
def InceptionResNetModel(width, height,lr):
    conv_base=InceptionResNetV2(include_top = False, weights = 'imagenet', input_tensor = None, input_shape = (width,height,3), pooling = None)
    model=models.Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(activation="relu", units=128))
    model.add(Dense(activation="sigmoid", units=1))
    print('Number of trainable weights before freezing the conv base:',len(model.trainable_weights))
    conv_base.trainable=False
    print('Number of trainable weights before freezing the conv base:', len(model.trainable_weights))
    model.compile(loss='binary_crossentropy',  optimizer=optimizers.Adam(lr=lr),  metrics=['acc'])
    return model