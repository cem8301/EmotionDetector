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
basePath = "/home/carolyn/Documents/Classes/DeepLearning/week14_finalProjects/FERPlus_data/code/myPics/"

imageList = [f for f in os.listdir(basePath) if os.path.isfile(basePath+f)]
imageList.sort()


# Load image and resize --------------------------------------------
for image in imageList:
   # New image
   print('--------------------------------------------------------')
   print('Name: ' + image)

   # Load
   pic = cv2.imread(basePath+image,cv2.IMREAD_GRAYSCALE)

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
      #norm = [float(i)/sum(ans)*100 for i in ans]
      for i in range(0,len(ans)):
         print(emotion_table[str(i)] + ': ' + str(round(ans[i]*100)) + '%')
   else:
      print(ans)
      print(sum(ans))
      print('No face')





