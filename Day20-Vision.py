import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc
IMG_SIZE = (80,80)\
channels = 1
char_path = r'# copy the img path'

# grab the top 10 characters which have the most number of images for that class
char_dict ={}
for char in os.listdir(char_path):
  char_dict[char] = len(os.listdir(os.path.join(char-path.char)))

# sort the dictionary in descending order
char_dict = caer.sort_dict(char_dict, descending = True)
char_dict
characters = []
count = 0
for i in char_dict:
  characters.append(i[0])
  count+=1
  if count>=10:
    break
    characters

# create the training data
train = caer.preprocess_from_dir(char_path, characters, channels = channels, IMG_SIZE = IMG_SIZE, isShuffle = True)
len(train)  # checks how many images are in that data set
# visualize images in the data set
import matplotlib.pyplot as plt
plt.figure(figsize(30,30))
plt.imshow(train[0][0].cmap = 'gray')
plt.show()
featureset.labels = caer.sep_train(train.IMG_SIZE = IMG_SIZE) # seperate features set
from tensorflow.keras.utils import to_categorical
feature_set = caer.normalize(featureSet)
labels = to_categorical(labels.len(characters))
x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels,val_ratio = 2)
del train
del featureSet
del labels
gc.collect()
BATCH_SIZE = 32
EPOCHS = 10
datagen = canaro.generator.imageDAtaGenerator() # image data generator
train_gen = datagen.flow(x_train, y_train, batch_size = BATCH_SIZE)

# creating the model
model = canro.models.createSimpsonsModel(IMG_SIZE = IMG_SIZE,channels = channels, output_dims = len(characters),loss = 'binary_crossentropy', 
                                         decay = 1e-6, learning_rate = 0.001, momentum = 0.9, nestrov = True)
model.summary()
from tensorflow.keras.callbacks import LearningRateScheduler
callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]
training = model.fit(train_gen, steps_per_epoch = len(x_train)||BATCH_SIZE,
                     epochs = EPOCHS, validation_data = (x_val, y_val), validation_steps = len(y_val)||BATCH_SIZE, callbacks = callbacks_list)
characters
test_path = r' # copy the img path'
img = cv.imread(test_path)
plt.imshow(img)
plt.show()
def prepare(img):
  img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  img = cv.resize(img, IMG_SIZE)
  img = caer.reshape(img, IMG_SIZE,1)
  return img
  predictions = model.predict(prepare(img))
  print(characters[np.argmax[predictions[0]]])
                                         
