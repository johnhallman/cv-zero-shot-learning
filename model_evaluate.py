import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import texttable as tt

import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Activation, Input, Dropout, Add
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam



def load_imgs(data_dir):
	animal_to_imgs = {}
	for animal_name in os.listdir(data_dir):
		animal_to_imgs[animal_name] = []
		animal_dir = data_dir + "/" + animal_name + "/"
		for img_name in os.listdir(animal_dir):
			img = plt.imread(animal_dir + img_name)
			animal_to_imgs[animal_name].append(img)
	return animal_to_imgs

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

def pred_class(model, img, classes):
	s = model.predict(np.expand_dims(img, axis=0))[0]
	probs = np.zeros(len(classes))
	for i, animal in enumerate(classes):
		probs[i] = np.prod(np.abs(s - 1.0 + animal_to_feat[animal]))
	return probs.argsort()[-1]

def pred_class_ham(model, img, classes):
	s = np.round(model.predict(np.expand_dims(img, axis=0))[0]).astype(int)
	score = np.zeros(len(classes))
	for i, animal in enumerate(classes):
		score[i] = np.sum(np.abs(s - animal_to_feat[animal]))
	return score.argsort()[0]

def pred_class_sum(model, img, classes):
	s = model.predict(np.expand_dims(img, axis=0))[0]
	probs = np.zeros(len(classes))
	for i, animal in enumerate(classes):
		probs[i] = np.sum(np.abs(s - animal_to_feat[animal]))
	return probs.argsort()[0]

def pred_class_harm(model, img, classes):
	eps = 1e-5
	s = model.predict(np.expand_dims(img, axis=0))[0]
	probs = np.zeros(len(classes))
	for i, animal in enumerate(classes):
		pos = np.sum(np.log(eps + np.abs(s - 1.0 + animal_to_feat[animal])))
		neg = np.sum(np.log(eps + np.abs(s - animal_to_feat[animal])))
		probs[i] = pos - neg
	return probs.argsort()[-1]

def predictions(model, classes, animal_to_images, pred_func=pred_class):
	y_pred, y_true = [], []
	for i, animal in enumerate(classes):
		for img in animal_to_images[animal]:
			y_true.append(i)
			y_pred.append(pred_func(model, img, classes))
	return y_pred, y_true

def pred_features(model, img):
	return np.round(model.predict(np.expand_dims(img, axis=0))[0]).astype(int)

def feature_preds(model, classes, animal_to_images):
	y_pred, y_true = [], []
	for animal in classes:
		for img in animal_to_images[animal]:
			y_true.append(animal_to_feat[animal])
			y_pred.append(pred_features(model, img))
	return y_pred, y_true

def draw_table(scores, predicates):
	table = tt.Texttable()
	table.set_cols_align(["l", "r", "l", "r", "l", "r", "l", "r", "l", "r"])
	table.set_cols_valign(["m", "m", "m", "m", "m", "m", "m", "m", "m", "m"])
	table.set_cols_width([8, 8, 8, 8, 8, 8, 8, 8, 8, 8])
	header = ["Feature", "Score", "Feature", "Score", "Feature", "Score", "Feature", "Score", "Feature", "Score"]
	rows = [header]
	for i in range(0, len(scores), 5):
	    temp = []
	    for j in range(5): 
	        temp.append(predicates[i + j])
	        temp.append(round(scores[i + j], 3))
	    rows.append(temp)
	table.add_rows(rows)
	print(table.draw())


if __name__=="__main__":

	print("Load data")
	animal_to_imgs = load_imgs("images_128x128")
	animal_to_feat, id_to_name, name_to_id, train_classes, test_classes = load_info()
	all_classes = train_classes + test_classes
	predicate_file = pd.read_csv("predicates.txt", header=None)
	predicates = []
	for line in predicate_file.iloc[:,0]: predicates.append(line.split()[-1])

	print("Load model 2")
	model2 = load_model("models/final-model-90.hdf5")


	"""
	print("frequencies!!!")
	freqs = np.zeros(len(predicates))
	N = 0
	for animal, animal_imgs in animal_to_imgs.items():
		N += len(animal_imgs)
		for i, label in enumerate(animal_to_feat[animal]):
			freqs[i] += label * len(animal_imgs)
	freqs = freqs / N
	#draw_table(freqs, predicates)


	print("irregularities!")
	irregularities = np.zeros(len(test_classes))
	for i, animal in enumerate(test_classes):
		irregularities[i] = - np.log(np.abs(1.0 - animal_to_feat[animal] - freqs)).sum()
		print(animal + ": {}".format(irregularities[i]))

	print("Load model 1")
	model1 = load_model("models/model_1.h5")

	print("Load model 2")
	model2 = load_model("models/final-model-90.hdf5")

	print("Compute test feature accuracy for 1")
	y_pred, y_test = feature_preds(model1, test_classes, animal_to_imgs)
	feat_accuracy = np.mean(np.abs(np.array(y_pred) + np.array(y_test) - 1), axis=0)
	draw_table(feat_accuracy, predicates)

	print("Compute test feature accuracy for 2")
	y_pred, y_test = feature_preds(model2, test_classes, animal_to_imgs)
	feat_accuracy = np.mean(np.abs(np.array(y_pred) + np.array(y_test) - 1), axis=0)
	draw_table(feat_accuracy, predicates)


	print("Distribution of animals in train classes")
	train_dist = np.zeros(len(train_classes))
	test_dist = np.zeros(len(test_classes))
	for i, animal in enumerate(train_classes): train_dist[i] = len(animal_to_imgs[animal])
	for i, animal in enumerate(test_classes): test_dist[i] = len(animal_to_imgs[animal])

	train_dist = train_dist / train_dist.sum()
	test_dist = test_dist / test_dist.sum()

	print("Train distribution:")
	for i, animal in enumerate(train_classes):
		print(animal + ": {}".format(train_dist[i]))
	print("Test distribution:")
	for i, animal in enumerate(test_classes):
		print(animal + ": {}".format(test_dist[i]))

	#print("Compute harmonic probability test class score")
	#y_pred, y_test = predictions(model1, train_classes, animal_to_imgs, pred_func=pred_class_harm)
	#print(np.mean(np.array(y_pred) == np.array(y_test)))


	print("Compute test feature accuracy for 1")
	y_pred, y_test = feature_preds(model1, test_classes, animal_to_imgs)
	print(np.mean(np.array(y_pred) == np.array(y_test)))

	print("Compute train feature accuracy for 1")
	y_pred, y_test = feature_preds(model1, train_classes, animal_to_imgs)
	print(np.mean(np.array(y_pred) == np.array(y_test)))

	#print("Compute harmonic probability test class score")
	#y_pred, y_test = predictions(model1, train_classes, animal_to_imgs, pred_func=pred_class_harm)
	#print(np.mean(np.array(y_pred) == np.array(y_test)))
	"""

