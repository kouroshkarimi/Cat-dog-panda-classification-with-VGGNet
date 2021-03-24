# IMPORT THE NECESSARY PACKAGES
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# LOADING DATA
print("[INFO] loading images...")
data = []
labels = []

path, dirs, files = next(os.walk("datasets")) # FILES IN OUR DATASET

for i in range(len(files)):


	img = cv2.imread("datasets/" + files[i]) # READ IMAGES
	img = cv2.resize(img, (32,32)).flatten()	# RESIZE AND FLATTERN IMAGE
	data.append(img)	# APPENDS TO DATA

	label = files[i] # READ LABEL OF IMAGE

	if label.startswith("c"):
		labels.append("cat")
	elif label.startswith("d"):
		labels.append("dog")
	elif label.startswith("p"):
		labels.append("panda")


print("Loading complete.")


# SCALE ROW PIXELS TO 0 TO 1 AND STORE THEM IN NP ARRAY
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# SPLIT DATA TO TRAIN AND TEST
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 42)

# BINARIZING OUR LABELS TO [1 0 0] = CAT , [0 1 0] = DOG , [0 0 1] = PANDA
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# define the 3072-1024-512-3 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape = (3072,), activation = 'sigmoid'))
model.add(Dense(512, activation = "sigmoid"))
model.add(Dense(len(lb.classes_), activation = "softmax"))

# IITIALIZE LEARNING RATE AND NUMBER OF EPOCHS
INIT_LR = 0.01
EPOCHS = 60

# WE INIT OUR OPTIMIZER AND OUR COMPILE
opt = SGD(lr = INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])# binary_crossentropy for 2 clas

# train the neural network
print("[INFO] Training network...")
H = model.fit(x = trainX, y = trainY, validation_data = (testX, testY), epochs=EPOCHS, batch_size = 32)
print("[INFO] Training complete.")

# EVALUATING OUR NUERAL NETWORK
print("[INFO] Evaluating network...")
predictions = model.predict(x = testX, batch_size = 32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save("myModel.h5")