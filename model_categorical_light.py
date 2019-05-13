# Final Project 
# Carolyn Mason
# Deep Learning
# 4-18-19

import keras
import sys, os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import regularizers
from keras.utils.np_utils import to_categorical
import csv
from itertools import islice
from math import ceil
import numpy as np
from keras import regularizers


# Set up Test/ Train Data ------------------------------------------

base_dir = '/home/carolyn/Documents/Classes/DeepLearning/week14_finalProjects/FERPlus_data/data'
data = '/fer2013/datalight.csv'
fer_path = base_dir+data

# List of labels
emotion_table = {'0' : 'anger',     # 4953
                 '1' : 'happy',     # 8989
                 '2' : 'sad',       # 6077
                 '3' : 'neutral'}   # 6198

# Setup important information for the model
num_classes = len(emotion_table)
batch_size = 128
epochs = 25

# Read csv
count = 0
train_data = []
train_labels = []
test_data = []
test_labels = []

with open(fer_path,'r') as csvfile:
    dataNum = len(csvfile.readlines())-1
    trainNum = ceil(dataNum*0.6)

with open(fer_path,'r') as csvfile:
    fer_rows = csv.reader(csvfile, delimiter=',')
    for row in islice(fer_rows, 1, None):
         if(count < trainNum):
            train_data.append([float(s)/255. for s in row[1].split(' ')])
            train_labels.append(row[0])
         else:
            test_data.append([float(s)/255. for s in row[1].split(' ')])
            test_labels.append(row[0])      
         count+=1

# Settle data
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

# Labels
train_labels = to_categorical(train_labels,num_classes)
test_labels = to_categorical(test_labels,num_classes)

# Reshaping
img_rows, img_cols = 48, 48
train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


# Building the network ---------------------------------------------

# Model
def createAndRun(llambda):
   model = models.Sequential()
   model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Conv2D(64, (3, 3), activation='relu'))
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Conv2D(128, (3, 3), activation='relu'))
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Dropout(0.25))
   model.add(layers.Conv2D(128, (3, 3), activation='relu'))
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Flatten())
   model.add(layers.Dropout(0.25))
   model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(llambda), activation='relu'))
   model.add(layers.Dropout(0.4))
   model.add(layers.Dense(num_classes, activation='sigmoid'))
   
   # Print summary
   model.summary()
   
   # Compilation
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
   
   # Data pre-processing 
   history = model.fit(train_data, train_labels,
                       batch_size= batch_size,
                       epochs=epochs,
                       verbose=1, 
                       validation_data=(test_data, test_labels))
   
   
   # Save model after training
   model.save('facial_expressions_l2_'+str(llambda)+'.h5')
   
   return history

# Create and run model ---------------------------------------

history1 = createAndRun(0.001)
history2 = createAndRun(0.0005)
history3 = createAndRun(0.0001)

all_history = {'0.001':history1,'0.005':history2,'0.0001':history3}

# Create Plots ----------------------------------------------

color = {'0.001':'r','0.005':'b','0.0001':'g'}

for val in color.keys():
   acc = all_history[val].history['acc']
   val_acc = all_history[val].history['val_acc']
   plt.plot(range(1,epochs+1), acc, 'o'+color[val], label='Training acc: '+val)
   plt.plot(range(1,epochs+1), val_acc, color[val], label='Validation acc: '+val)

plt.title('Training and validation accuracy')
plt.legend()

plt.savefig('Training_and_validation_accuracy.png')
plt.figure()



for val in color.keys():
   loss = all_history[val].history['loss']
   val_loss = all_history[val].history['val_loss']
   plt.plot(range(1,epochs+1), loss, 'o'+color[val], label='Training loss: '+val)
   plt.plot(range(1,epochs+1), val_loss, color[val], label='Validation loss: '+val)

plt.title('Training and validation loss')
plt.legend()

plt.savefig('Training_and_validation_loss.png')
plt.show()

# Results
for val in color.keys():
   val_acc = all_history[val].history['val_acc'][-1]
   print('L2 value: ' +str(val) + ', Accuracy: ' + str(val_acc))




