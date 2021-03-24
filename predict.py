
# IMPORT NECESSARY PACHAGES
from tensorflow.keras.models import load_model
import pickle
import cv2
import numpy as np

img = cv2.imread("panda2.jpg")
output = img.copy()
img = cv2.resize(img, (64,64))
img = img.astype("float")/255.0
print(img.shape)
#img = img.flatten()
img = img.reshape(1,64,64,3)


# LOAD OUR MODEL
print("[INFO] Loading network...")
model = load_model("myCnnModel.h5")
predic = model.predict(img)

# SHOW RESULT
max_arg = np.argmax(predic)
max_val = np.max(predic)
print(max_arg)

if max_arg == 0:
	str_prd = "cat"
elif max_arg == 1:
	str_prd = "dog"
elif max_arg == 2:
	str_prd = "panda"

text =  "{}: {:.2f}%".format(str_prd, max_val * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(0, 255, 255), 2)
cv2.imshow("output", output)
cv2.waitKey(0)