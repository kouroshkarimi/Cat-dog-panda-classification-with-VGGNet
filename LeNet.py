# IMPORT PACKAGES
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten
from tensorflow.keras import backend as K

# DEFINE CLASS OF OUR NETWORK

class LeNet:
	@staticmethod
	# n_channels =  channels of images (here we have 1 in mnist)
	# (imgRow, imgCols) = width and height of image 
	# n_class = number of output class
	# activation = our activation function (not for last layer)
	def build(n_channels, imgRows, imgCols, n_class,
			  activation = 'relu', w_path = None):
		
		if K.image_data_format() == "channels_first":
			in_shape = (n_channels, imgRows, imgCols)
		else:
			in_shape = (imgRows, imgCols, n_channels)

		# DEFINE OUR MODEL	
		model = Sequential([Conv2D(20, 5, padding ='same', activation = activation, input_shape = in_shape),
						   MaxPooling2D(pool_size = (2,2)),
						   Conv2D(50, 5, padding ='same', activation = activation),
						   MaxPooling2D(pool_size = (2,2)),
						   Flatten(),
						   Dense(500, activation = activation),
						   Dense(n_class, activation = 'softmax')])

		# IF WE WANT TO USE OUR SAVED WEIGHTS
		if w_path is not None:
			model.load_weights(w_path)


		return model

