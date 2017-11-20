from dataSetOld import *
from deepLearnerOld import *
import os
import numpy as np

file_dir = "./data_stage2/"
models = {} 

# Go through all the weather files and prepare data sets. 
for file in os.listdir(file_dir):
    if file.endswith(".txt"):
    	modelName = file.split('.txt')[0]
    	X_set = []
    	Y_set = []    	
    	for line in open(file_dir+file):
    		l = line.replace('\n', '')
    		X_set.append(l.split('===')[0].split())
    		Y_set.append(l.split('===')[1].split())
    	models[modelName] = (X_set, Y_set)

# Now iterate through all the models, and determine the best input-output representation.
for key in models:	
	print '-'*50
	print 'Designing model: ***', key, '***'
	print '(DataSet)'

	# analyse data and represent as needed, count some items we need to know.
	sub_dataset = DataSet(np.asarray(models[key][0]), np.asarray(models[key][1]))
	sub_dataset.vocab = sub_dataset.countSymbols()
	print sub_dataset.vocab
	sub_dataset.max_len = sub_dataset.countMaxLen()
	X, Y = sub_dataset.representData(models[key][0], models[key][1])
	
	X_train, X_test, Y_train, Y_test = sub_dataset.shuffleAndSplit(X, Y)
	print '-'*50		

	# now choose and compile a deep learning model, train it and save the outputs.
	print '(DeepLearner)'	
	sub_deep = DeepLearner('LSTM', layers=4, hidden_size=20, epochs=2000)
	model = sub_deep.designModel(sub_dataset.max_len, sub_dataset.vocab, sub_dataset.dimensionality)
	sub_deep.compileModel(model)		
	sub_deep.trainVerbose(X_train, Y_train, X_test, Y_test, sub_dataset.indices_char, sub_dataset.dimensionality, './out_stage2/'+key+'-weights.h5', './out_stage2/'+key+'-out.txt')	
	
		
