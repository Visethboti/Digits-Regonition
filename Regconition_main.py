# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 18:49:14 2020

@author: Hister
"""

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import time



# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# 32, num feature detectors 3 by 3
classifier.add(Convolution2D(20, 5, 5, input_shape = (128, 128, 3), activation = 'relu'))

# Step 2 - Max Pooling
# Usually we use 2 by 2
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(40, 5, 5, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
# Hidden layer
classifier.add(Dense(output_dim = 40, activation = 'relu'))

# Output layer
classifier.add(Dense(output_dim = 3, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)# rescale the value to between 0 and 1

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 3,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = 3,
                                            class_mode = 'categorical')



starttime = time.time()

classifier.fit_generator(training_set,
                         epochs = 100,
                         validation_data = test_set,
                         workers = 6)

print('That took {} seconds'.format(time.time() - starttime))




"""
import numpy as np
from keras.preprocessing import image

y_pred = []
for i in range(0,21):
    test_image = image.load_img('dataset/folder_test/test_{}.png'.format(i), target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    result = classifier.predict_classes(np.expand_dims(test_image, axis = 0))
    y_pred.append(result)
"""


import numpy as np
from keras.preprocessing import image

y_pred = []
for i in range(0,1):
    test_image = image.load_img('dataset/folder_test2/test_{}.png'.format(i), target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    result = classifier.predict_classes(np.expand_dims(test_image, axis = 0))
    y_pred.append(result)





"""
# Using steps_per_epoch and validation_steps instand of samples per epoch
starttime = time.time()
classifier.fit_generator(training_set,
                         steps_per_epoch = 250, 
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 63, 
                         workers = 6)
print('That took {} seconds'.format(time.time() - starttime))
"""


"""
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
"""

