from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding, Flatten, Reshape, Dropout
import time
import numpy as np

class DeepLearner:

	def __init__(self, modelString, layers=2, batch_size=32, epochs=50, hidden_size=128, embedding_size=1000):

		self.modelString = modelString
		self.model = ''
		self.layers = layers		
		self.batch_size = batch_size
		self.epochs = epochs
		self.hidden_size = hidden_size
		self.embedding_size = embedding_size		
		
		
	def designModel(self, max_len, vocab, dimensionality):		
		# Based on data representation, choose the best model to design. 
		
		print max_len
		if dimensionality=='3D':
			self.designSeq2Seq(max_len, vocab)
		# two paths for 2D models, recurrent or not.	
		elif self.modelString=='MLP':
			self.designMLP(max_len)	
		else:
			self.designRNN(max_len)		
		
		
		
	def designSeq2Seq(self, max_len, vocab):
		# Puts together a model that learns to map an input sequence to an output sequence. Requires 3D input matrices.
		print 'Designing a(n) sequence-to-sequence', self.modelString, 'with', self.layers, 'layers,', self.hidden_size, 'hidden nodes, and softmax activation.'
		
		RNN = recurrent.LSTM
		if self.modelString=='LSTM':
			RNN = recurrent.LSTM	
		elif self.modelString=='RNN':
			RNN = recurrent.SimpleRNN
		elif self.modelString=='GRU':
			RNN = recurrent.GRU
		
		self.model = Sequential()
		self.model.add(RNN(self.hidden_size, input_shape=(max_len, len(vocab))))
		self.model.add(RepeatVector(max_len))
		for _ in range(self.layers):
		    self.model.add(RNN(self.hidden_size, return_sequences=True))
		self.model.add(TimeDistributed(Dense(len(vocab))))
		self.model.add(Activation('softmax'))

		self.model.summary()		            	  
		
		return self.model


	def designRNN(self, max_len):
		# Puts together a model that learns to map an input sequence to a single output value. Requires 2D input matrices.
		print 'Designing a(n)', self.modelString, 'with', self.layers, 'layers,', self.hidden_size, 'hidden nodes, and softmax activation.'
		
		RNN = recurrent.LSTM
		if self.modelString=='LSTM':
			RNN = recurrent.LSTM	
		elif self.modelString=='RNN':
			RNN = recurrent.SimpleRNN
		elif self.modelString=='GRU':
			RNN = recurrent.GRU	
		
		self.model = Sequential()
		self.model.add(Embedding(self.embedding_size, self.hidden_size, input_length=max_len, dropout=0.2))
		self.model.add(RNN(self.hidden_size, dropout_W=0.2, dropout_U=0.2))  # can replace LSTM by RNN or GRU
		self.model.add(RepeatVector(max_len))
		for _ in range(self.layers):
		    self.model.add(RNN(self.hidden_size, return_sequences=True))		
		self.model.add(Flatten())
		self.model.add(Dense(1))
		self.model.add(Activation('sigmoid'))

		self.model.summary()		            	  
		
		return self.model

		
	def designMLP(self, max_len):
		# Puts a model together that learns to map an input sequence to an output sequence. Requires 3D input matrices.
		print 'Designing a(n) ', self.modelString, 'with', self.layers, 'layers,', self.hidden_size, 'hidden nodes, and softmax activation.'
		
		self.model = Sequential()
		self.model.add(Dense(self.hidden_size, input_shape=(max_len,)))
		self.model.add(Activation('relu'))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(1))
		self.model.add(Activation('sigmoid'))	

		self.model.summary()		            	  
		
		return self.model		
		
		
	def compileModel(self, model, loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']):
		# Compiles the model ready for training or testing with the chosen optimisation and training parameters. 
		print 'Model is ready to train (or test), using', loss, ',', optimizer, 'optimisation, and evaluating', metrics, '.'
		self.model.compile(loss=loss,
        	      optimizer=optimizer,
            	  metrics=metrics)	
            	  
            	  
	def train(self, X_train, Y_train, X_val, Y_val, output_file='out.txt'):
		# Actually trains for a given number of epochs.
		
		#print X_train
		print('Started training at:', (time.strftime("%d/%m/%Y")), (time.strftime("%H:%M:%S")))	
	
		for iteration in range(1, self.epochs):
		    print()
		    print('-' * 50)
		    print('Iteration', iteration)
		    self.model.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=1, validation_data=(X_val, Y_val))
		    score, acc = self.model.evaluate(X_val, Y_val, batch_size=self.batch_size)                           
		    json_string = self.model.to_json()
		    self.model.save_weights(output_file, overwrite=True)		    
                          
		    print('Test score:', score)
		    print('Test accuracy:', acc)     	           	  			
            	  
		print('Finished at:', (time.strftime("%d/%m/%Y")), (time.strftime("%H:%M:%S")))            	  


	def decode3D(self, X, indices_char, calc_argmax=True):
		# this decodes 3D items. 
		if calc_argmax:
			X = X.argmax(axis=-1)
		return ' '.join(indices_char[x] for x in X)	
		
	
	def decode2D(self, X_item, indices_char):
		# this decodes 2D X items. 
		new_X = []
		for x in X_item:
			new = indices_char[int(x)]
			new_X.append(new)
		new_X = ' '.join(new_X).replace('mask_zeros', '')
		return ' '.join(new_X.split())			            	  


	def trainVerbose(self, X_train, Y_train, X_val, Y_val, indices_char, dimensionality, output_file='out.txt', log_file="log.txt"):
		if dimensionality=='3D':
			self.trainVerbose3D(X_train, Y_train, X_val, Y_val, indices_char, output_file, log_file)
		else:
			self.trainVerbose2D(X_train, Y_train, X_val, Y_val, indices_char, output_file)						

		            	  
	def trainVerbose2D(self, X_train, Y_train, X_val, Y_val, indices_char, output_file='out.txt'):
		# Actually trains for a given number of epochs and prints some example tests after each epoch.
		
		#print X_train
		print('Started training at:', (time.strftime("%d/%m/%Y")), (time.strftime("%H:%M:%S")))	
	
		for iteration in range(1, self.epochs):
		    print()
		    print('-' * 50)
		    print('Iteration', iteration)
		    self.model.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=1, validation_data=(X_val, Y_val))
		    score, acc = self.model.evaluate(X_val, Y_val, batch_size=self.batch_size)
                            
                            
		    # Select 10 samples from the validation set at random to visualise and inspect. 
		    for i in range(10):
		        ind = np.random.randint(0, len(X_val))
		        rowX, rowy = X_val[np.array([ind])], Y_val[np.array([ind])]
		        preds = self.model.predict_classes(rowX, verbose=0)
		        q = self.decode2D(rowX[0], indices_char)
		        correct = int(rowy[0])
		        guess = int(preds[0])
		        print()
		        print('Input vector: ', q)
		        print('Correct label: ', correct)
		        print('Predicted label: ' + str(guess) + ' (good)' if correct == guess else 'Predicted label: ' + str(guess) + ' (bad)' )
		        print('---')
		        json_string = self.model.to_json()
		        self.model.save_weights(output_file, overwrite=True)                             
                          
		    print('Test score:', score)
		    print('Test accuracy:', acc)     	           	  			
            	  
		print('Finished at:', (time.strftime("%d/%m/%Y")), (time.strftime("%H:%M:%S")))

            	  
	def trainVerbose3D(self, X_train, Y_train, X_val, Y_val, indices_char, output_file='out.h5', log_file="log.txt"):
		# Actually trains for a given number of epochs and prints some example tests after each epoch.
		
		log = open(log_file, 'w')
		log.close()
		log = open(log_file, 'a')		
		
		#print X_train
		print('Started training at:', (time.strftime("%d/%m/%Y")), (time.strftime("%H:%M:%S")))	
		log.writelines('Started training at: ' + str(time.strftime("%d/%m/%Y")) + " "+ str(time.strftime("%H:%M:%S")) + '\n')
		
	
		for iteration in range(1, self.epochs):
		    print()
		    log.writelines(' ' + "\n")
		    print('-' * 50)
		    log.writelines('-' * 50 + "\n")
		    print('Iteration', iteration)
		    log.writelines('Iteration ' + str(iteration) + "\n")
		    self.model.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=1, validation_data=(X_val, Y_val))
		    score, acc = self.model.evaluate(X_val, Y_val, batch_size=self.batch_size)
                            
                            
		    # Select 10 samples from the validation set at random to visualise and inspect. 
		    for i in range(10):
		        ind = np.random.randint(0, len(X_val))
		        rowX, rowy = X_val[np.array([ind])], Y_val[np.array([ind])]
		        preds = self.model.predict_classes(rowX, verbose=0)
		        q = self.decode3D(rowX[0], indices_char)
		        correct = self.decode3D(rowy[0], indices_char)
		        guess = self.decode3D(preds[0], indices_char, calc_argmax=False)
		        print()
		        log.writelines(' ' + "\n")		        
		        print('Input vector: ', q)
		        log.writelines('Input vector: ' + str(q) + "\n")
		        print('Correct label: ', correct)
		        log.writelines('Correct label: ' + str(correct) + "\n")  
		        print('Predicted label: ' + str(guess) + ' (good)' if correct == guess else 'Predicted label: ' + str(guess) + ' (bad)' )
		        log.writelines('Predicted label: ' + str(guess) + ' (good)' if correct == guess else 'Predicted label: ' + str(guess) + ' (bad)' + '\n')		        		        
		        print('---')
		        log.writelines('---' + "\n")		        
		        json_string = self.model.to_json()
		        self.model.save_weights(output_file, overwrite=True)                             
                          
		    print('Test score:', score)
		    log.writelines('Test score:' + str(score) + "\n")
		    print('Test accuracy:', acc)     	           	  			
		    log.writelines('Test accuracy:' + str(acc) + "\n")		    
            	  
		print('Finished at:', (time.strftime("%d/%m/%Y")), (time.strftime("%H:%M:%S")))
		log.writelines('Finished at: ' + str(time.strftime("%d/%m/%Y")) +" "+ str(time.strftime("%H:%M:%S")) + "\n")
		log.close()
	


	# needs to have methods from training and testing and for model definition
	# needs to have a method to load weights from an existing model.	
	


