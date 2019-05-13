import pandas as pd
from sklearn.utils import shuffle

# Grab csv
base_dir = '/home/carolyn/Documents/Classes/DeepLearning/week14_finalProjects/FERPlus_data/data'
data = '/fer2013/fer2013.csv'
fer_path = base_dir+data

df = pd.read_csv(fer_path, 
                 header=0, 
                 names=['Emotion', 'data', 'etc'])

# Simplify the emotions
l1 = df.loc[df['Emotion'] == 0]  # Angry
l2 = df.loc[df['Emotion'] == 3]  # Happy
l3 = df.loc[df['Emotion'] == 4]  # Sad
l4 = df.loc[df['Emotion'] == 6]  # Nuetral

light = pd.concat([l1,l2,l3,l4])

# Fix values
light.Emotion.loc[(light['Emotion'] == 3)] = 1
light.Emotion.loc[(light['Emotion'] == 4)] = 2
light.Emotion.loc[(light['Emotion'] == 6)] = 3

# re-shuffle
light = shuffle(light)

# Save csv
light.to_csv(base_dir+'light.csv')
