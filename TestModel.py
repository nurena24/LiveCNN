from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import backend as K
import os
#https://stackoverflow.com/questions/52270177/how-to-use-predict-generator-on-new-images-keras
Classifier_Name = 'TrexBronto' #should be the name of the classifier parent directory
MODEL_NAME='TrexBronto-12-0.90'

Width=150
Height=150

#Classifier_Name = 'RocketPopsicle' #should be the name of the classifier parent directory

print('loading classifier')
classifier = load_model(Classifier_Name+'/'+MODEL_NAME+'.h5')
print('Classifier Loaded!')

#Create the DataGenerator which loads the files from the directory
#batch size is set to 1 because we want to predict 1 file at a time.
test_dir = Classifier_Name+'/dataset/pred_set/'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(Width, Height),
        color_mode="rgb",
        shuffle = False,
        class_mode=None,
        batch_size=1)
filenames = test_generator.filenames
nb_samples = len(filenames)

#Create List of Filenames in the directory so we can show the image with the prediction result
pred_img_dir=test_dir+'all_classes/'
images=[]
for filename in os.listdir(pred_img_dir):
   images.append(filename)
#using Test Generator with batch size of 1, create an array prediction of all the files in the folder nb_samples.
predict = classifier.predict_generator(test_generator,steps = nb_samples)

#this is where we plot the results. Number of prediction/images must be a multiple of 10 to keep everything even on the plot
rows=nb_samples/10
i=0
text_labels=[]
plt.figure()
for x,y in zip(predict,images):
        plt.subplot(rows, 10, i+1)
        if x >0.5:
                print('trex')
                img=mpimg.imread(pred_img_dir+y)
                imgplot = plt.imshow(img)
                text_labels.append('trex')#trex#dog

        else:
                print('bronto')
                img = mpimg.imread(pred_img_dir + y)
                imgplot = plt.imshow(img)
                text_labels.append('bronto')#bronto#cat


        plt.title('This is a ' + text_labels[i])
        i += 1
plt.show()