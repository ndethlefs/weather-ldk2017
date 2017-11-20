import numpy as np
from keras.preprocessing import sequence
from keras.engine.training import slice_X

class DataSet:

	def __init__(self, X_set, Y_set, max_len=0):
		self.X_set = X_set
		self.Y_set = Y_set		
		self.max_len = max_len
	
		self.text = 'mask_zeros '
		self.vocab = set()		  
		self.char_indices = {}
		self.indices_char = {}
		self.dimensionality = ''
		self.outputs = 1		
		if not hasattr(Y_set[0], '__iter__'):
			self.outputs = len(set(Y_set))			


	def representData(self, X_set, Y_set):
		# Find out what representation is needed, 2D or 3D. 	
		sample_X = X_set[0]
		sample_Y = Y_set[0]		
		
		if hasattr(sample_Y, '__iter__'):
			print('Use a 3D representation to model sequence-to-sequence task... (encoding.)')		
			self.dimensionality = '3D'			
			X, Y = self.representDataAs3D(X_set, Y_set)
		else:
			print('Use a 2D representation for classification task... (encoding.)')		
			self.dimensionality ='2D'			
			X, Y = self.representDataAs2D(X_set, Y_set)	

		return X, Y	


	def representDataAs2DNumeric(self, X_set, Y_set):
		# Use this for tasks that classify a single output value.
		X_2D = np.asarray(X_set)
		Y_2D = np.asarray(Y_set)		
		
		print 'Representing data X and Y with shape:', X_2D.shape
		new_X = sequence.pad_sequences(new_X, maxlen=self.max_len)		
		
		# normally works without manipulation of Y, as Y is just a single array of 
		# possible output values. Hence only re-represent X. 
				
		return new_X, Y_2D


	def representDataAs2D(self, X_set, Y_set):
		# Use this for tasks that classify a single output value.
		X_2D = np.asarray(X_set)
		Y_2D = np.asarray(Y_set)		
		
		# Get mapping dicts for input representations.
		self.char_indices = dict((c, i) for i, c in enumerate(self.vocab))
		self.indices_char = dict((i, c) for i, c in enumerate(self.vocab)) 
		print self.char_indices		
		
		print 'Representing data X and Y with shape:', X_2D.shape
		
		new_X = self.encode2DIntegerX(X_2D)
		new_X = sequence.pad_sequences(new_X, maxlen=self.max_len)		
		
		# normally works without manipulation of Y, as Y is just a single array of 
		# possible output values. Hence only re-represent X. 
				
		return new_X, Y_2D
		
		
	def representDataAs3D(self, X_set, Y_set):
		# Use this for sequence-to-sequence learning with 3D methods below.		
		X_3D = np.zeros((len(X_set), self.max_len, len(self.vocab)), dtype=np.bool)
		Y_3D = np.zeros((len(Y_set), self.max_len, len(self.vocab)), dtype=np.bool)		
		
		# Get mapping dicts for input representations.
		self.char_indices = dict((c, i) for i, c in enumerate(self.vocab))
		self.indices_char = dict((i, c) for i, c in enumerate(self.vocab))  				
		
		print 'Representing data X and Y with shape:', X_3D.shape
		
		new_X = self.encode3DBooleanX(X_set)
		new_Y = self.encode3DBooleanX(Y_set)	
		print 'Padding all sequences to max_len:', self.max_len
		new_X = sequence.pad_sequences(new_X, maxlen=self.max_len)
		new_Y = sequence.pad_sequences(new_Y, maxlen=self.max_len)		
		
		return new_X, new_Y
		

	def encode2DIntegerX(self, X_set):		
		# encode X into a 2D matrix / array of arrays.
		for i, item in enumerate(X_set):
			for j, jtem in enumerate(item):
				item[j] = self.char_indices[jtem]
			X_set[i] = item
		return X_set	
			

	def encode3DBooleanX(self, X_set):
		# encode X into a boolean matrix
		for i, item in enumerate(X_set):
			X1 = np.zeros((self.max_len, len(self.vocab)))
			for j, c in enumerate(item):
				X1[j, self.char_indices[c]] = 1
			X_set[i] = X1			
		return X_set	


	def encode3DBooleanY(self, Y_set):
		# encode Y into a boolean matrix
		for i, item in enumerate(Y_set):
			Y1 = np.zeros((self.max_len, len(self.vocab)))
			for j, c in enumerate(item):
				Y1[j, char_indices[c]] = 1
			Y[i] = Y1	
		return Y_set
	
	
	
	def decode3D(self, X, calc_argmax=True):
		# this decodes 3D items. 
		if calc_argmax:
			X = X.argmax(axis=-1)
		return ' '.join(indices_char[x] for x in X)	
		
	
	def decode2D(X_item, model):
		# this decodes 2D X items. 
		new_X = []
		for x in X_item:
			new = indices_char[int(x)]
			new_X.append(new)
		new_X = ' '.join(new_X).replace('mask_zeros', '')
		return ' '.join(new_X.split())		
		
		
	def countSymbols(self):
		# find all the different types of symbols in the input and output. 
	
		for x in self.X_set:
			self.text = self.text + ' '.join(x) + ' '

		if hasattr(self.Y_set[0], '__iter__'):
			for y in self.Y_set:
				self.text = self.text + ' '.join(y) + ' '
		
		chars = set(self.text.split())
		vocab = list(chars)
		vocab.insert(0, 'mask_zeros')		
		print 'Found', len(vocab), 'unique vocabulary items.' 			
			
		return vocab			


	def countMaxLen(self):
		# find the max len of an input or output sequence so we can pad all sequences. 
		for x in self.X_set:
			if len(x) > self.max_len:
				self.max_len = len(x)
		if hasattr(self.Y_set[0], '__iter__'):				
			for y in self.Y_set:
				if len(y) > self.max_len:
					self.max_len = len(y)		
		return self.max_len
		
		
	def shuffleAndSplit(self, X, Y):		
		
		# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits	
		indices = np.arange(len(Y))
		X = X[indices]
		Y = Y[indices]
		
		# Explicitly set apart 10% for validation data that we never train over		
		split_at = len(X) - len(X) / 10
		(X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
		(Y_train, Y_val) = (Y[:split_at], Y[split_at:])		
		
		print 'Shuffled and split dataset into', len(X_train), 'training instances and', len(X_val), 'test instances.'
		
		return X_train, X_val, Y_train, Y_val
			
		
		
		