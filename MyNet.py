# IMPORT PACKAGES
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.layers import Activation, Flatten
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

# DEFINE CLASS OF OUR NETWORK

class MyNet:
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
		weight_decay = 1e-4
		model = Sequential()
		model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=in_shape))
		model.add(Activation('elu'))
		model.add(BatchNormalization())
		model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('elu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.2))
		
		model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('elu'))
		model.add(BatchNormalization())
		model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('elu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.3))
		
		model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('elu'))
		model.add(BatchNormalization())
		model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('elu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.4))
		
		model.add(Flatten())
		model.add(Dense(n_channels, activation='softmax'))

		


		return model

