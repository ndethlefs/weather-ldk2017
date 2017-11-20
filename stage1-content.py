from dataSet import *
from deepLearner import *

'''
Deep learning model for the first stage of weather forecasting: from a set of weather 
records, decide for each whether it is likely to appear in a weather forecast or not. 

Using an LSTM with 4 layers and 50 hidden nodes, this achieves a performance of
97.8% after 50 training epochs (or sooner). 
'''

X_set = []
Y_set = []
input_file = './weather-events2-new.txt'
output_file = input_file.split('.txt')[0] + '-stage1-weights.h5'

# Read data from a file and populate X and Y. 
for line in open(input_file, 'r'):
	data = line.split('===')[1]
	entries = data.split('|')
	label = 0	
	l = line.split('===')[2].replace('\n', '').split()	
	for e in entries:
		if '.type' in e:
			X_set.append(e.split())
			if e.split()[0] in l:
				label = 1
			else:
				label = 0	
			Y_set.append(label)	
	
	
### Initiate a new class, weather_data, get the number of unique symbols, lengths of input and output sequences. 
weather_data = DataSet(X_set, Y_set)
weather_data.vocab = weather_data.countSymbols()
weather_data.max_len = weather_data.countMaxLen()

### Format data so we can pass it to a deep learner. 
X, Y = weather_data.representData(X_set, Y_set) 
X_train, X_test, Y_train, Y_test = weather_data.shuffleAndSplit(X, Y)

# Initialise a new deep learner class, valid models are GRU, LSTM, RNN or NN. 
deep = DeepLearner('LSTM', 4, hidden_size=50, epochs=50)

# Design a sequence-to-sequence learner, compile and train it.
model = deep.designModel(weather_data.max_len, weather_data.vocab, weather_data.dimensionality)
deep.compileModel(model)
deep.trainVerbose(X_train, Y_train, X_test, Y_test, weather_data.indices_char, weather_data.dimensionality, output_file)