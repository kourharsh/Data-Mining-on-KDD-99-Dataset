"""
Created on Fri Nov  1 18:52:31 2019

@author: Adil
"""
import time
from random import seed
from random import randrange
from math import sqrt
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.metrics import confusion_matrix


#split the data for into subsets
def cross_validation_split(dataset, partitions):
	dataset_with_partitions = list()
	original = list(dataset)
	#determine the size of fold
	fold_size = int(len(dataset) / partitions)
	for value in range(partitions):
		partition=list()
		#keep adding tuples until fold size is achieved
		while len(partition) < fold_size:
			position = randrange(len(original))
			new_instance=original.pop(position)
			partition.append(new_instance)
		dataset_with_partitions.append(partition)
	return dataset_with_partitions



#method the get the accuracy
def get_accuracy(actual, predicted):
	rightanswers = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			rightanswers= rightanswers+1
	accuracy=rightanswers / float(len(actual)) * 100.0
	return accuracy

#Calculate the Euclidean Distance between two tuples/ instances
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1) - 1):
		temp= (row1[i] - row2[i]) *(row1[i] - row2[i])
		distance=distance+temp
	return sqrt(distance)


#KNN Algorithm
def knn(data_train, data_test, no_neighbors):
	results = list()
	#for each instance calculate distance from each instance in train data
	for instance in data_test:
		distances = list()
		for train_instance in data_train:
			dist = euclidean_distance(instance, train_instance)
			distances.append((train_instance, dist))
		#sort distances in ascending order
		distances.sort(key=lambda tup: tup[1])
		neighbors = list()
		for i in range(no_neighbors):
			neighbors_distance=distances[i][0]
			neighbors.append(neighbors_distance)
		predictions = [row[-1] for row in neighbors]
		#choose the majority vote class as prediction
		prediction = max(set(predictions), key=predictions.count)
		results.append(prediction)
	return results

def KNNalgorithm(ns):
	#read data from dataset csv using pandas
	data = pd.read_csv(r'subsetkdd99_proc.csv')
	dataset = data.values.tolist()
	#Number of partitions or folds for cross validation split
	partitions =2

	#code for choosing the value of k
	num = int(sqrt(len(dataset)) / 2)
	if (num % 2 == 0):
		neighbors = num - 1
	else:
		neighbors = num

	#k value set to the k passed
	neighbors = ns

	#split the data using cross validation and get the folds of data
	folds = cross_validation_split(dataset, partitions)
	results = list()
	#run knn for each fold
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_data = list()
		#set the last value to None to predict it using knn
		for tuple in fold:
			temp = list(tuple)
			test_data.append(temp)
			temp[-1] = None
		#call knn to get predicted values
		predicted = knn(train_set, test_data, neighbors)
		#the actual values for each fold
		actual = [row[-1] for row in fold]

		#code to display confusion matrix
		classes=['back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap', 'ipsweep', 'land', 'loadmodule','multihop', 'neptune', 'nmap', 'normal', 'perl', 'phf', 'pod', 'portsweep', 'rootkit', 'satan', 'smurf', 'spy', 'teardrop', 'warezclient', 'warezmaster']
		actual1=[classes[int(i)] for i in actual]
		predicted1 = [classes[int(i)] for i in predicted]
		# confusion matrix visual
		mat = confusion_matrix(predicted1, actual1, labels=classes)
		df_cm = pd.DataFrame(mat, index=[i for i in classes], columns=[i for i in classes])
		plt.figure(figsize=(10, 5))
		sn.heatmap(df_cm, annot=True)
		print(metrics.confusion_matrix(actual1, predicted1))

		#calculating the accuracy
		accuracy = get_accuracy(actual, predicted)

		# confusion matrix metrics like precision, recall, f1 score
		print(metrics.classification_report(actual1, predicted1, digits=3))
		results.append(accuracy)
	print('The percentage of Correctly Classified Results for each fold are : %s' % results)
	print('The percentage of Accuracy is: %.3f%%' % (sum(results) / float(len(results))))
	return results


# call algorithm for value k=2
seed(1)
t0 = time.clock()

#since in this project, we have considered all the dimensions, KNN gives the best results when the k value is small like k=1 or k=3

print(KNNalgorithm(1))
t1 = time.clock() - t0
print("Time to complete the algorithm: ", t1 - t0)


#Future improvements
#---Feature selection/reduction to avoid curse if Dimensionality