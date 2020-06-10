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
from pylab import *

# Importing the dataset
X = []
y = []

#pil_im = Image.open('3_2.png')
pil_im = Image.open('3_1.png').convert('L')
out = pil_im.resize((128,128))
#imshow(out)
#show()
imArray = array(out).flatten()

X.append(imArray)
y.append(1)

pil_im = Image.open('3_2.png').convert('L')
out = pil_im.resize((128,128))
imArray = array(out).flatten()
X.append(imArray)
y.append(1)

pil_im = Image.open('3_3.png').convert('L')
out = pil_im.resize((128,128))
imArray = array(out).flatten()
X.append(imArray)
y.append(1)

pil_im = Image.open('3_4.png').convert('L')
out = pil_im.resize((128,128))
imArray = array(out).flatten()
X.append(imArray)
y.append(1)

pil_im = Image.open('4_1.png').convert('L')
out = pil_im.resize((128,128))
imArray = array(out).flatten()
X.append(imArray)
y.append(0)

pil_im = Image.open('4_2.png').convert('L')
out = pil_im.resize((128,128))
imArray = array(out).flatten()
X.append(imArray)
y.append(0)


pil_im = Image.open('4_3.png').convert('L')
out = pil_im.resize((128,128))
imArray = array(out).flatten()
X_test.append(imArray)

pil_im = Image.open('nig.png').convert('L')
out = pil_im.resize((128,128))
imArray = array(out).flatten()
X_test.append(imArray)

pil_im = Image.open('3_5.png').convert('L')
out = pil_im.resize((128,128))
imArray = array(out).flatten()
X_test.append(imArray)

pil_im = Image.open('3_6.png').convert('L')
out = pil_im.resize((128,128))
imArray = array(out).flatten()
X_test.append(imArray)
# Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
X_train = X
y_train = y
X_test = []
y_test = []
X_test.append(imArray)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


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