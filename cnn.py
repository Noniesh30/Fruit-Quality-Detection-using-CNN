from keras.models import Sequential
import tensorflow as tf
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random 
from numpy import *
from PIL import Image
import os
train_mat={}
currentdir = os.getcwd()
filenames = os.listdir(os.path.join(currentdir, "train"))
for file in filenames:
  with open(os.path.join("train",file), 'rb') as image:
    train_mat[file] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
trainimage = list(train_mat.values())
for i in range(len(trainimage)):
     trainimage[i]= cv2.resize(trainimage[i], (224,224))
for i in range(len(trainimage)):
    trainimage[i]= cv2.cvtColor(trainimage[i], cv2.COLOR_BGR2RGB)
train_label=[]
for key,val in train_mat.items():
   train_label.append(key.split("/")[0])
for i in range(len(trainimage)):
    trainimage[i]=list(trainimage[i])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(trainimage, train_label, test_size = 0.2, random_state = 4)
batch_size = 16
nb_classes =4
nb_epochs = 5
img_rows, img_columns = 224, 224
img_channel = 3
nb_filters = 3
nb_pool = 2
nb_conv = 3
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(3, (3,3), padding='same', activation=tf.nn.relu,
                           input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(3, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(4,  activation=tf.nn.softmax)
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, batch_size = batch_size, epochs = nb_epochs, verbose = 1, validation_data = (X_test, y_test))
y_pred= model.predict(X_test)
total=len(y_pred)
sum=0
for i in range(len(y_pred)):
    x1=y_pred[i]
    x2=y_test[i]
    if(x1[0]==x2[0]):
        sum+=1 
acc=(sum/total)*100
print("Accuracy of the model :",acc)