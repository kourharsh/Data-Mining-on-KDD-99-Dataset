"""
Created on Fri Nov 1 18:26:28 2019

@author: Nikunj
"""
from sklearn import metrics
from random import randrange
import pandas as pd
from random import seed
import time
import matplotlib.pyplot as plt
#import confusion_matrix
import seaborn as sn


numberOffolds = 2

maximum_depth = 40

minimum_size = 1

#t0 = time.time()
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


def final_split_function(node, maximum_depth, minimum_size, depth):
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
		node['left'] = getSplitFunction(left)
		final_split_function(node['left'], maximum_depth, minimum_size, depth+1)
	
	if len(right) <= minimum_size:
		node['right'] = into_the_terminal(right)
	else:
		node['right'] = getSplitFunction(right)
		final_split_function(node['right'], maximum_depth, minimum_size, depth+1)
  
      
def getSplitFunction(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
        
		for row in dataset:
            
			groups = test_split_function(index, row[index], dataset)
			gini = gini_function_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
                
	return {'index':b_index, 'value':b_value, 'groups':b_groups}


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


def final_decision_tree(train, test, maximum_depth, minimum_size):
    tree = getSplitFunction(train)
    final_split_function(tree, maximum_depth, minimum_size, 1)
    predictions = list()
    for row in test:
        prediction = prediction_function(tree, row)
        predictions.append(prediction)
    return(predictions)

t0 = time.time()
seed(1)

fname = 'subsetkdd99_proc.csv'
dataset = data_from_pandas(fname)

 
outcomes = decision_tree_algorithm(dataset, final_decision_tree, numberOffolds, maximum_depth, minimum_size)
print('Correctly Classified Results Per Fold', outcomes)
print('Correctly Classified Result of Decision Tree: ' , (sum(outcomes)/float(len(outcomes))))
t1 = time.time()
print("Time elapsed: ", t1 - t0)