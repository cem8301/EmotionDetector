# Final Project 
# Carolyn Mason
# Deep Learning
# 4-18-19

import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing import image
import os


# Load Model --------------------------------------------------------
model = load_model('/home/carolyn/Documents/Classes/DeepLearning/week14_finalProjects/FERPlus_data/code/facial_expressions.h5')

# List of labels
emotion_table = {'0' : 'anger',     # 4953
                 '1' : 'happy',     # 8989
                 '2' : 'sad',       # 6077
                 '3' : 'neutral'}   # 6198


# Setup ------------------------------------------------------------
basePath = "/home/carolyn/Documents/Classes/DeepLearning/week14_finalProjects/FERPlus_data/data/light/"


for emotion in emotion_table.values():
   # Reset count
   corCount = 0 
   totCount = 0
   
   # Find all iamges in each emotion directory
   imageList = [f for f in os.listdir(basePath+emotion) if os.path.isfile(basePath+emotion+'/'+f)]
   imageList.sort()

   # Loop over all images
   for image in imageList:
      # Load
      pic = cv2.imread(basePath+emotion+'/'+image,cv2.IMREAD_GRAYSCALE)

      # Resize and Trim = 32x32 pixels
      sz = 48
      r = sz / pic.shape[0]
      dim = (int(pic.shape[1] * r),sz)
      resized = cv2.resize(pic, dim, interpolation = cv2.INTER_AREA)
      img = resized[0:sz,0:sz]

      # Option too save for reference
      #cv2.imwrite(basePath+"/cropped.jpg", img)

      # Preprocess the image into a 4D tensor
      img2 = np.expand_dims(img, axis=0)
      img_tensor = np.expand_dims(img2, axis=3)

      # Predict 
      ans = model.predict(img_tensor)[0]
      if (sum(ans) > 0.02 ):
         if(emotion == emotion_table[str(np.argmax([ans]))]):
            corCount += 1

      totCount += 1
   
   print(emotion + '--------------------------------------------')
   print(str(corCount)+'/'+str(totCount))
         






