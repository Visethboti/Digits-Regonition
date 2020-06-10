# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:50:42 2020

@author: Hister
"""

# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from pylab import array
from pylab import *


# Importing the dataset
X_train = []
y_train = []
X_test = []
y_test = []


pil_im = Image.open('dataset/folder_test/test_2.png').convert('LA').resize((128,128))
imArray = array(pil_im).flatten()
width, height = pil_im.size 

left = width / 4
top = height / 4
right = width / 4
bottom = height / 4

im1 = pil_im.crop((left, top, right, bottom)) 
im1.show()

imshow(pil_im)
show()



# Import Image to train
for _ in range(0, 1):
    for i in range(1, 31):
        pil_im = Image.open('dataset/folder_3/3_{}.png'.format(i)).convert('LA').resize((128,128))
        imArray = array(pil_im).flatten()
        X_train.append(imArray)
        y_train.append(3)

    for i in range(1, 31):
        pil_im = Image.open('dataset/folder_4/4_{}.png'.format(i)).convert('LA').resize((128,128))
        imArray = array(pil_im).flatten()
        X_train.append(imArray)
        y_train.append(4)
    
    for i in range(1, 26):
        pil_im = Image.open('dataset/folder_5/5_{}.png'.format(i)).convert('LA').resize((128,128))
        imArray = array(pil_im).flatten()
        X_train.append(imArray)
        y_train.append(5)

# test set
X_test = []
for i in range(1, 5):
    pil_im = Image.open('dataset/folder_test/test_{}.png'.format(i)).convert('LA').resize((128,128))
    imArray = array(pil_im).flatten()
    X_test.append(imArray)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
classifier.fit(X_train, y_train)

    

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)


import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
im = cv2.imread('folder_test/test_2.png.jpg')
bbox, label, conf = cv.detect_common_objects(im)
output_image = draw_bbox(im, bbox, label, conf)
plt.imshow(output_image)
plt.show()


"""

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

"""