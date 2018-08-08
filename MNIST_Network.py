from keras.layers import Input,Convolution2D as Conv2D, MaxPooling2D, concatenate
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Model, Sequential
from keras.utils import plot_model, np_utils
from keras.datasets import mnist
from keras import backend as K
K.set_image_dim_ordering('th')
import sys, numpy as np


def neural_network():
	"""
		Defines architecture of the model.

		This network has input layer of 28x28 images, followed by convoluted layer conv1, which outputs 28x28x32. 
		Conv1 layer is input to two convoluted layer conv2_1 and conv2_2 both of them having shape of 14x14x64.
		Conv2_1 is input to layer conv3_1 and conv2_2 is input to layer conv3_2. Conv3_1 and conv3_2 are merged to
		form layer conv3 and has shape of 7x7x512. Then layer Conv3 is connected to fully connected layer fc_1 having
		dimension of 1000. Layer fc_1 is then connected to fc_2 having dimension of 500. Evenutally there is a drop-
		out layer whose input are obtained form fc_2. Dropout layer drops about 20% of the randomly selected features
		from the feature map. This short of functionality helps netwrok to better generalize. Evenutally the dropout
		layer is connected to the output layer that returns probability of each class. 

		Linear rectified units (RelU) are used as activation function and for downsampling maxpooling is used.

		@returns: model 
	"""
	image_input = Input(shape=(1, 28, 28))

	conv1  = Conv2D(32, (3, 3), input_shape=(1, 28, 28), activation='relu', strides=(1, 1), padding='same')(image_input)
	conv1_pooling = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_pooling)
	conv2_1_pooling = MaxPooling2D(pool_size=(2, 2))(conv2_1)

	conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_pooling)
	conv2_2_pooling = MaxPooling2D(pool_size=(2, 2))(conv2_2)

	conv3_1 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv2_1_pooling)
	conv3_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv2_2_pooling)
	# Combine layer conv3_1 and conv3_2 to form conv3.
	conv3 = concatenate([conv3_1, conv3_2])

	dense_layer = Flatten()(conv3)
	fc_1 = Dense(1000, activation='relu')(dense_layer)
	fc_2 = Dense(500, activation='relu')(fc_1)
	drop2 = Dropout(0.2)(fc_2)
	out = Dense(10, activation='softmax')(fc_2)
	model = Model(inputs=image_input, outputs=out)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


def prepare_mnist_data():
	"""
		Prepare and pre-process the mnist dataset.

		Loads data from Keras datasets and processes it for the model training purpose.
		@returns: X_train, X_test, y_train, y_test
	"""
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	# reshape to be [samples][pixels][width][height]
	X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
	X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

	# normalize the data.
	X_train = X_train / 255
	X_test = X_test / 255

	# for output purpose convert categorical data to one hot encode data
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	return X_train, X_test, y_train, y_test


def main(epc, batch, model_name):
	train_data, test_data, train_labels, test_labels = prepare_mnist_data()

	# Load model and save the visualization of the model.
	model = neural_network()
	plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

	# Train the model using training set.
	model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=epc, batch_size=batch, verbose=1)

	# Evaluate model on the test set and print results
	scores = model.evaluate(test_data, test_labels, verbose=1)
	print("CNN Error: %.2f%%" % (100-scores[1]*100))

	# Save the trained model.
	model.save(model_name)


if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser(description = 'MNIST Neural Network Training Argument Parser')
	parser.add_argument('--epochs', type=int, default=10 ,help='Enter number of epochs')
	parser.add_argument('--batch_size', type=int, default=200 ,help='Enter size of each batch while traning the netwrok')
	parser.add_argument('--model_name', type=str, default='handwritten_digit_recog_model.h5', help='Enter the name for saving the model')
	args = parser.parse_args()
	main(args.epochs, args.batch_size, args.model_name)
