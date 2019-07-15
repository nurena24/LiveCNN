from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import backend as K
import os
import cv2
from grabscreen import grab_screen
import time
import numpy as np
from PIL import Image
from random import shuffle
#https://stackoverflow.com/questions/52270177/how-to-use-predict-generator-on-new-images-keras
#MODEL_NAME='TrexBrontoES15'
MODEL_NAME='RocketvPopsiclesES15'
Width=150
Height=150
print('loading classifier')
classifier = load_model(MODEL_NAME+'.h5')
print('Classifier Loaded!')
def load(img):
    #img = cv2.imread(filename)
    img = cv2.resize(img, (Width, Height))
    img = np.array(img).astype('float32') / 255
    img = np.expand_dims(img, axis=0)  # all this does is make a (150,150,3) tuple into a (1,150,150,3) tuple
    return img
#using Test Generator with batch size of 1, create an array prediction of all the files in the folder nb_samples.
#predict = classifier.predict_generator(test_generator,steps = nb_samples)

#this is where we plot the results. Number of prediction/images must be a multiple of 10 to keep everything even on the plot
last_time = time.time()
while (True):
    screen = grab_screen(region=(400, 100, 1920, 1080))  # -10 and -20 for offset of toolbars when playing mnousebot
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    #resize for prediction
    screenPred = cv2.resize(screen, (Width, Height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    print('loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    # load the image
    image = load(screenPred)
    prediction = classifier.predict(image)
    #print(prediction)
    if prediction > 0.5:
        result = 'Popscicle'
        #result = str(round((prediction * 100), 2)) + 'Popscicle'
        #result = 'trex'
    else:
        result = 'Rocket'
       # result = ((1 - prediction) * 100) + 'Rocket'
        # result = 'bronto'
    cv2.putText(screen,result,(10,500), font, 8, (255,255,155), 8, cv2.LINE_AA)
    cv2.imshow('window', cv2.resize(screen, (1920, 1080)))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
        #I need to see the image before training...then i need to see the image before prediction to see if they match...both normalized??
