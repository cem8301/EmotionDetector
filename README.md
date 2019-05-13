# EmotionDetector

# Sample Results
![Alt text](https://github.com/cem8301/EmotionDetector/blob/master/splash.png?raw=true "Python HTML Version")


Case Study in Recognition of Emotion from Images
Carolyn Mason
4/18/2019
CSCI E-89 Deep Learning

Problem:
It can take a lot of time to sort through and choose images to share with friends and family. It can also be hard to come back to an album years later and find a specific sets of pictures. Deep learning tools can help speed up this process and pinpoint images of a specific style that you are searching for. This tool is an emotion detector, allowing the user to pick the top ‘X’ images from their photo album. The user can choose from angry, happy, sad, and neutral. 

Data Sets:
Data from Kaggle: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
 
Featured Technology:
This project demonstrates python, Keras, and CNN (Convelutional Neural Networks). 

High Level Overview of Steps:
1) Download the data from Kaggle
2) Setup test and train data sets
3) Build the CNN
4) Iterate until happy with the model results
5) Run on your own images!

Challenges:
Top challenges were overfitting, category result imbalance, and importance of cropping test images. Full images cannot be used to predict emotion, since they are too complex. The user my crop the image down close to the sides of the face. 

Results:
The final accuracy of the model is 66.5% for predicting four emotions (angry, happy, sad, neutral). The model is best at determining happy emotion and worst at determining sad emotion.

URLs for YouTube recordings:
2 minute: https://youtu.be/nrLpXucMcLM 
15 minute: https://youtu.be/mJR3sblItPs 

References: 
Zoran’s class notes
Kaggle Data set: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
Deep Facial Expression Recognition: A Survey: https://arxiv.org/pdf/1804.08348.pdf 
Variety of papers on facial recognition: https://paperswithcode.com/task/facial-expression-recognition 
Deep Learning For Smile Recognition: https://arxiv.org/pdf/1602.00172v2.pdf 




