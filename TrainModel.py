from models import inception,InceptionResNetModel
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint,TensorBoard
import os
import time
#log_dir="logs/{}".format(time())
#Settable Parameters!!
Width = 150
Height = 150
nClasses = 2
learningRate = 1e-4
EPOCHS = 40
PATIENCE = 15
Classifier_Name = 'TrexBronto' #should be the name of the classifier parent directory
#Classifier_Name = 'RocketPopsicle' #should be the name of the classifier parent directory
#Settable Parameters!!
if not os.path.exists('logs/'+Classifier_Name):
        os.makedirs('logs/'+Classifier_Name)

classifier = InceptionResNetModel(Width,Height,learningRate)
# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
mc = ModelCheckpoint(Classifier_Name+'/'+Classifier_Name+'-{epoch:02d}-{val_acc:.2f}.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
tensorboard_callback = TensorBoard(log_dir='logs/'+Classifier_Name, update_freq=1000)
#Create Image Manipulator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
#load images into training_set
training_set = train_datagen.flow_from_directory(Classifier_Name+'/dataset/training_set', target_size = (Width, Height), batch_size = 32, class_mode = 'binary')
test_set = test_datagen.flow_from_directory(Classifier_Name+'/dataset/test_set',  target_size = (Width, Height), batch_size = 32, class_mode = 'binary')

nb_train_samples = len(training_set.filenames)#number of training images
nb_test_samples = len(test_set.filenames)#number of test images
classifier.fit_generator(training_set,
                         samples_per_epoch = nb_train_samples*2,#how many times does the network look at all the training samples? here it is doubled. Training Samples x2#8000
                         epochs = EPOCHS,
                         validation_data = test_set,
                         nb_val_samples = nb_test_samples*2,#1000 how many test images are t here? here it is doubled
                         verbose=0,
                         callbacks=[es,mc,tensorboard_callback])

#tensorboard --logdir=foo:C:/Users/Nick/PycharmProjects/DogCatCNN/logs
#conda install tensorflow-gpu
#conda install keras-gpu