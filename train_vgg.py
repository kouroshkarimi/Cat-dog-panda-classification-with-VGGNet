

# IMPORT NECESSARY PACKAGES
from SmallVGGNet import SmallVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

# LOADING DATA
print("[INFO] loading images...")
data = []
labels = []

path, dirs, files = next(os.walk("datasets")) # FILES IN OUR DATASET

for i in range(len(files)):


	img = cv2.imread("datasets/" + files[i]) # READ IMAGES
	img = cv2.resize(img, (64,64))	# RESIZE AND FLATTERN IMAGE
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

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
# initialize our VGG-like Convolutional Neural Network
model = SmallVGGNet.build(width=64, height=64, depth=3,
	classes=len(lb.classes_))

# initialize our initial learning rate, # of epochs to train for,
# and batch size
INIT_LR = 0.01
EPOCHS = 75
BS = 32

# initialize the model and optimizer (you'll want to use
# binary_crossentropy for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (SmallVGGNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


# Saving the model
model.save("myCnnModel.h5")
print("[INFO] Model saved.")

