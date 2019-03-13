import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Activation, Input, Dropout, Add
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


# takes a data directory and returns a dictionary mapping animal names to a list of images
def load_imgs(data_dir):
	animal_to_imgs = {}
	for animal_name in os.listdir(data_dir):
		animal_to_imgs[animal_name] = []
		animal_dir = data_dir + "/" + animal_name + "/"
		for img_name in os.listdir(animal_dir):
			img = plt.imread(animal_dir + img_name)
			animal_to_imgs[animal_name].append(img)
	return animal_to_imgs
	


# takes a directory and loads relevant txt files into dictionaries
def load_info():
	df_classes = pd.read_csv("classes.txt", header=None) 
	df_predicate_matrix = pd.read_csv("predicate-matrix-binary.txt", header=None)
	df_test_classes = pd.read_csv("testclasses.txt", header=None)
	df_train_classes = pd.read_csv("trainclasses.txt", header=None)

	animal_to_feat = {}
	id_to_name, name_to_id = {}, {}
	for i, c in enumerate(df_classes[0]):
		c_name = c.split()[1]
		id_to_name[i] = c_name
		name_to_id[c_name] = i
		animal_to_feat[c_name] = np.array([int(binary) for binary in df_predicate_matrix.iloc[i, 0].split()])

	train_classes, test_classes = [], []
	for c in df_train_classes[0]: train_classes.append(c.split()[0])
	for c in df_test_classes[0]: test_classes.append(c.split()[0])

	return animal_to_feat, id_to_name, name_to_id, train_classes, test_classes



# takes a subset of classes, imgs, and feature mappings and returns data in x, y format (numpy array)
def classes_to_data(classes, animal_to_imgs, animal_to_feat):
		x_data, y_data, l_data = [], [], []
		for animal in classes:
			feats = animal_to_feat[animal].tolist()
			x_data += animal_to_imgs[animal]
			y_data += len(animal_to_imgs[animal]) * [feats]
			l_data += len(animal_to_imgs[animal]) * [animal]
		return x_data, y_data, l_data



# compute accuracy of model on given classes
def compute_score(model, classes):
	score = []
	for animal in classes: 
		for img in animal_to_imgs[animal]: # loop over all images corresponding to classes
			s = model.predict(np.expand_dims(img, axis=0))[0] # feature probabilities
			probs = np.zeros(len(classes))
			for i, animal in enumerate(classes): # probability for every class
				probs[i] = np.prod(np.abs(s - 1.0 + animal_to_feat[animal]))
			score.append((animal == classes[probs.argsort()[-1]])) # is argmax == true label?
	return np.mean(score)



# returns skeleton of model of type 1
def model_1(output_dim=85):
	model = Sequential()
	model.add(Conv2D(16, kernel_size=5, activation='relu', padding='same', input_shape=(128,128,3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
	model.add(Dropout(0.1))
	model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))    # shape == 32 x 32 here
	model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
	model.add(Dropout(0.3))
	model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))    # shape == 16 x 16 here
	model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(output_dim, activation='sigmoid')) # 85 = number of features
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model



# returns skeleton of model of type 1
def model_2(output_dim=85):
	def RegBlock(x, depth, kernel_size, strides=(1, 1)):
		y = Conv2D(depth, kernel_size, strides=strides, padding='same')(x)
		y = BatchNormalization()(y)
		return Activation('relu')(y)

	def ResidueBlock(x, depth=32, kernel_size=(5, 5)):
		y = BatchNormalization()(x)
		y = Activation('relu')(y)
		y = RegBlock(y, depth, kernel_size)
		y = Conv2D(depth, kernel_size=kernel_size, padding='same')(y)
		return Add()([x, y])

	a = Input(shape=(128, 128, 3))
	x = RegBlock(a, 64, (5, 5), strides=(2, 2))  # shape == 64 x 64
	for i in range(6): x = ResidueBlock(x, depth=64, kernel_size=(3, 3))
	x = RegBlock(x, 128, (3, 3), strides=(2, 2))  # shape == 32 x 32
	for i in range(6): x = ResidueBlock(x, depth=128, kernel_size=(3, 3))
	x = RegBlock(x, 256, (3, 3), strides=(2, 2))  # shape = 16 x 16
	for i in range(6): x = ResidueBlock(x, depth=256, kernel_size=(3, 3))
	x = RegBlock(x, 32, (3, 3))
	x = Flatten()(x)
	x = Dense(output_dim)(x)
	b = Activation('sigmoid')(x)

	model = Model(inputs=a, outputs=b)
	opt = Adam(lr=0.0003)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model



# takes data and trains model of given type, runs cross validation, then returns results
def build_model(create_model, train_classes, test_classes, animal_to_imgs, animal_to_feat, k=5, epochs=10):
	n, subset_size = len(train_classes), int(len(train_classes) / k)
	split_classes = [train_classes[x:x + subset_size] for x in range(0, n, subset_size)]

	# cross validation
	cross_val_accuracy = []
	"""
	for i, cross_test_classes in enumerate(split_classes):
		if i > 0: break
		print("Cross-validation for round {} of {}".format(i+1, k))
		cross_train_classes = [c for c in train_classes if c not in cross_test_classes]
		x_train, y_train, l_train = classes_to_data(cross_train_classes, animal_to_imgs, animal_to_feat)
		x_test, y_test, l_test = classes_to_data(cross_test_classes, animal_to_imgs, animal_to_feat)

		model = create_model()
		#checkpoint = ModelCheckpoint(filepath="cross-model-{epoch:02d}.hdf5", period=10)
		model.fit(np.array(x_train), np.array(y_train), epochs=epochs, verbose=2) # callbacks=[checkpoint]
		current_score = compute_score(model, cross_test_classes)
		print("Current round cross-validation accuracy: {}".format(current_score))
		cross_val_accuracy.append(current_score)
	"""

	# full model
	print("Training full model")
	x_train, y_train, l_train = classes_to_data(train_classes, animal_to_imgs, animal_to_feat)
	x_test, y_test, l_test = classes_to_data(test_classes, animal_to_imgs, animal_to_feat)

	model = create_model()
	#checkpoint = ModelCheckpoint(filepath="full-model-{epoch:02d}.hdf5", period=10)
	model.fit(np.array(x_train), np.array(y_train), epochs=epochs, verbose=2) # callbacks=[checkpoint]
	test_accuracy = compute_score(model, test_classes)
	return model, test_accuracy, np.mean(cross_val_accuracy)



# train models and store in model directory
if __name__ == "__main__":

	data_dir = sys.argv[1] # directory of all the data files
	assert os.path.isdir(data_dir)

	print("Loading files and data")
	animal_to_imgs = load_imgs(data_dir)
	animal_to_feat, id_to_name, name_to_id, train_classes, test_classes = load_info()
	all_classes = train_classes + test_classes

	# train and store model 1

	print("Training and testing model 1 for 25 epochs")
	model, acc, cross_acc = build_model(model_1, train_classes, test_classes, animal_to_imgs, animal_to_feat, epochs=25)
	model_1_filename = "dropout_model_1.h5"
	model.save(model_1_filename)
	print("Model 1: acc, cross_acc = {}, {}".format(acc, cross_acc))
	print("Model 1 saved to " + model_1_filename)

	"""
	# train and store model 2
	print("Training and testing model 2 for 100 epochs")
	model, acc, cross_acc = build_model(model_2, train_classes, test_classes, animal_to_imgs, animal_to_feat, epochs=100)
	model_2_filename = "final_model_2.h5"
	#model.save(model_2_filename)
	print("Model 2: acc, cross_acc = {}, {}".format(acc, cross_acc))
	#print("Model 2 saved to " + model_2_filename)
	"""





