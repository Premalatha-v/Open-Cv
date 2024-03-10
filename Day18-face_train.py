import os
import cv2 as cv
import numpy as np
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfeild', 'Madonna', 'Mindy Kaling']
p = []
for i in os.listdir(r' # copy and paste the img path'):
  p.append(i)
  print(p)
  DIR = r' # paste the img path '
  haar_cascade = cv.cascadeClassifier('haar_face.xml')

# Essentially image arrays of faces
features = []
labels = []
def create_train():
  for person in people

  # grab path for the person 
  path = os.path.join(DIR, person)
  label = people.index(person)

#going to loop over every image in that folder
for img in os.listdir(path):
img_path = os.path.join(path,img)

#read image from that path
img_array = cv.imread(img_path)

# convert this image into gray
gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
faces_rect = haar_cascade.detectMultiScale(gray, ScaleFactor = 1.1, minNeighbors = 4)
for(x, y, w, h) in faces_rect:
  faces_roi = gray[y:y+h, x:x+w]
  features.append(faces_roi)
  labels.append(label)
  create_train()
  print('Training done __________')
  features = np.array(features, dtype = 'object')
  labels = np.array(labels)
  face_recognizer = cv.face_LBPHFaceRecognizer_create()
  face_recognizer.train(features, labels)
# save features and labels list
np.save('Features.npy', features)
np.save('Labels.npy', labels)
