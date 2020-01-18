# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:46:48 2019

@author: Nikunj
"""
from sklearn import metrics
from random import randrange
import pandas as pd
from random import seed
from math import sqrt
import time
import matplotlib.pyplot as plt
#import confusion_matrix
import seaborn as sn


numberOffolds = 2

maximum_depth = 40

minimum_size = 1

sample_size = 6.0

#Number_of_trees = [10]

def data_from_pandas(fname):
    dataset_pandas = pd.read_csv(fname)
    
    dataset = dataset_pandas.values.tolist()
    
    return dataset


def split_method(dataset, numberOffolds):
    dataset_split = []
    dataset_copy  = list(dataset)
    dataset_list  = dataset_copy
    fold_size = int(len(dataset) / numberOffolds)
    
    for a in range(numberOffolds):
        
        fold = list()        
        while len(fold) < fold_size:
          
            index = randrange(len(dataset_list))
            fold.append(dataset_list.pop(index))
            
        dataset_split.append(fold)
  
    return dataset_split

def metrics_of_accuracy(actual, predicted):
	correction = 0    
    
	for a in range(len(actual)):
		if actual[a] == predicted[a]:
			correction += 1
            
	return correction / float(len(actual)) * 100.0

def decision_tree_algorithm(dataset, algorithm, numberOffolds, *args):
   fold1 = split_method(dataset, numberOffolds)
   folds = fold1
   outcomes = list()
   
   for fold in folds:
     train_set1 = list(folds)
     train_set = train_set1
     train_set.remove(fold)
     train_set = sum(train_set, [])
     test_set = list()
     
     for row in fold:
        row_dum = list(row)
        row_copy = row_dum
        test_set.append(row_copy)
        row_copy[-1] = None
        
     predicted1 = algorithm(train_set, test_set, *args)
     predicted = predicted1
     actual = [row[-1] for row in fold]
     accuracy = metrics_of_accuracy(actual, predicted)
     outcomes.append(accuracy)
     classes=['back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap', 'ipsweep', 'land', 'loadmodule','multihop', 'neptune', 'nmap', 'normal', 'perl', 'phf', 'pod', 'portsweep', 'rootkit', 'satan', 'smurf', 'spy', 'teardrop', 'warezclient', 'warezmaster']
     actual1=[classes[int(i)] for i in actual]
     predicted1 = [classes[int(i)] for i in predicted]
     mat=metrics.confusion_matrix(predicted1, actual1, labels=classes)
     df_cm = pd.DataFrame(mat, index=[i for i in classes],columns=[i for i in classes])
     plt.figure(figsize=(10, 5))
     sn.heatmap(df_cm, annot=True)    
     print(metrics.confusion_matrix(actual, predicted))
     print(metrics.classification_report(actual, predicted))
     
   return outcomes

def test_split_function(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
        
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
            
	return left, right

def gini_function_index(groups, classes):
	n_instances = float(sum([len(group) for group in groups]))
	gini = 0.0
	for group in groups:
        
		size = float(len(group))
		if size == 0:
			continue
		score = 0.0
		for class_val in classes:
            
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
            
		gini += (1.0 - score) * (size / n_instances)
        
	return gini

def into_the_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

def final_split_function(node, maximum_depth, minimum_size, n_features, depth):
	left, right = node['groups']
	del(node['groups'])
	if not left or not right:
		node['left'] = node['right'] = into_the_terminal(left + right)
		return
    
	if depth >= maximum_depth:
		node['left'], node['right'] = into_the_terminal(left), into_the_terminal(right)
		return
	
	if len(left) <= minimum_size:
		node['left'] = into_the_terminal(left)
	else:
		node['left'] = getSplitFunction(left, n_features)
		final_split_function(node['left'], maximum_depth, minimum_size, n_features, depth+1)
	
	if len(right) <= minimum_size:
		node['right'] = into_the_terminal(right)
	else:
		node['right'] = getSplitFunction(right, n_features)
		final_split_function(node['right'], maximum_depth, minimum_size, n_features, depth+1)
        
def getSplitFunction(dataset, n_features):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	features = list()
	while len(features) < n_features:
        
		index = randrange(len(dataset[0])-1)
		if index not in features:
			features.append(index)   
            
	for index in features:
        
         for row in dataset:           
            groups = test_split_function(index, row[index], dataset)
            gini = gini_function_index(groups, class_values)
            if gini < b_score:
               b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

def final_decision_tree(train, max_depth, min_size, n_features):    
	root = getSplitFunction(train, n_features)
	final_split_function(root, max_depth, min_size, n_features, 1)
	return root

def prediction_function(node, row):
	if row[node['index']] < node['value']:
        
		if isinstance(node['left'], dict):
			return prediction_function(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return prediction_function(node['right'], row)
		else:
			return node['right']
    
def subsample(dataset, ratio):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
        
		index = randrange(len(dataset))
		sample.append(dataset[index])
        
	return sample

def bagging_predict(trees, row):
	predictions = [prediction_function(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)

def random_forest(train, test, max_depth, min_size, sample_size, number_of_trees, n_features):
	trees = list()
	for i in range(number_of_trees):
        
		sample = subsample(train, sample_size)
		tree = final_decision_tree(sample, max_depth, min_size, n_features)
		trees.append(tree)
        
	predictions = [bagging_predict(trees, row) for row in test]
    
	return(predictions)
    
t0 = time.time()
seed(1)

fname = 'subsetkdd99_proc.csv'

dataset = data_from_pandas(fname)

n_features = int(sqrt(len(dataset[0])-1))

input_number_of_trees = 3

for number_of_trees in [input_number_of_trees]:
    outcomes = decision_tree_algorithm(dataset, random_forest, numberOffolds, maximum_depth, minimum_size,sample_size, number_of_trees, n_features)
    print('Correctly Classified Results Per Fold', outcomes)
    print('Correctly Classified Result of Random Forest: ' , (sum(outcomes)/float(len(outcomes))))
    t1 = time.time()
    print("Time elapsed: ", t1 - t0)