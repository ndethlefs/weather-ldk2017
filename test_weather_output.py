from dataSetOld import *
from deepLearnerOld import *
import os
import numpy as np

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'
    highlight = '\033[94m' 
    bold = '\033[1m'

def getNumeric(entry):
					
	e = []
	for i, item in enumerate(entry):
		if 'id' in item:
			id = int(item.split(':')[1])
			e.append(id)
		if 'min' in item:
			min = int(entry[i+1])
			e.append(min)			
		if 'max' in item:
			max = int(entry[i+1])
			e.append(max)			
		if 'mean' in item:
			mean = int(entry[i+1])
			e.append(mean)			
	return e
	
def getNumericBucket(entry):
					
	e = []
	for i, item in enumerate(entry):
		if 'id' in item:
			id = int(item.split(':')[1])
			e.append(id)
		if 'min' in item:
			min = int(entry[i+1])
			e.append(min)			
		if 'max' in item:
			max = int(entry[i+1])
			e.append(max)			
		if 'mean' in item:
			mean = int(entry[i+1])
			e.append(mean)			
		if 'mode-bucket' in item:
			range = entry[i+1]
			r1 = range.split('-')[0]
			r2 = range.split('-')[1]
			e.append(int(r1))
			e.append(int(r2))			
	return e	
	
	
def getSymbolicWind(entry):
					
	e = []
	dict = {'S': 0, 'SW': 1, 'SSE': 2, 'WSW': 3, 'ESE': 4, 'E': 5, 'W': 6, 'SE': 7, 'NE': 8, 'SSW': 9, 'NNE': 10, 'WNW': 11, 'N': 12, 'NNW': 13, 'ENE': 14, 'NW': 15}
	
#	['S', 'SW', 'SSE', 'WSW', 'ESE', 'E', 'W', 'SE', 'NE', 'SSW', 'NNE', 'WNW', 'N', 'NNW', 'ENE', 'NW']
	for i, item in enumerate(entry):
		if 'id' in item:
			id = int(item.split(':')[1])
			e.append(id)
		if 'mode' in item:
			e.append(entry[i+1])
	return e


def getSymbolic(entry):
					
	e = []
	dict = {'--': 0, 'SChc': 1, 'Chc': 2, 'Def': 3, 'Lkly': 4}
	for i, item in enumerate(entry):
		if 'id' in item:
			id = int(item.split(':')[1])
			e.append(id)
		if 'mode' in item:	
			e.append(entry[i+1])
	return e
	
	
def getNumericS3(entry):
					
	e = []
	for i, item in enumerate(entry):
		if 'id' in item:
			id = int(item.split(':')[1])
			e.append(id)
		if 'min' in item:
			min = int(entry[i+1])
			e.append(min)			
		if 'max' in item:
			max = int(entry[i+1])
			e.append(max)			
		if 'mean' in item:
			mean = int(entry[i+1])
			e.append(mean)	
		if 'time' in item:
			t1 = entry[i+1].split('-')[0]
			t2 = entry[i+1].split('-')[1]
			e.append(t1)
			e.append(t2)			
	return e
	
	
def getNumericBucketS3(entry):
					
	e = []
	for i, item in enumerate(entry):
		if 'id' in item:
			id = int(item.split(':')[1])
			e.append(id)
		if 'min' in item:
			min = int(entry[i+1])
			e.append(min)			
		if 'max' in item:
			max = int(entry[i+1])
			e.append(max)			
		if 'mean' in item:
			mean = int(entry[i+1])
			e.append(mean)			
		if 'mode-bucket' in item:
			range = entry[i+1]
			r1 = range.split('-')[0]
			r2 = range.split('-')[1]
			e.append(int(r1))
			e.append(int(r2))		
		if 'time' in item:
			t1 = entry[i+1].split('-')[0]
			t2 = entry[i+1].split('-')[1]
			e.append(t1)
			e.append(t2)				
	return e	
	
	
def getSymbolicWindS3(entry):
					
	e = []
	dict = {'S': 0, 'SW': 1, 'SSE': 2, 'WSW': 3, 'ESE': 4, 'E': 5, 'W': 6, 'SE': 7, 'NE': 8, 'SSW': 9, 'NNE': 10, 'WNW': 11, 'N': 12, 'NNW': 13, 'ENE': 14, 'NW': 15}
	
#	['S', 'SW', 'SSE', 'WSW', 'ESE', 'E', 'W', 'SE', 'NE', 'SSW', 'NNE', 'WNW', 'N', 'NNW', 'ENE', 'NW']
	for i, item in enumerate(entry):
		if 'id' in item:
			id = int(item.split(':')[1])
			e.append(id)
		if 'mode' in item:
			e.append(entry[i+1])
		if 'time' in item:
			t1 = entry[i+1].split('-')[0]
			t2 = entry[i+1].split('-')[1]
			e.append(t1)
			e.append(t2)			
	return e


def getSymbolicS3(entry):
					
	e = []
	dict = {'--': 0, 'SChc': 1, 'Chc': 2, 'Def': 3, 'Lkly': 4}
	for i, item in enumerate(entry):
		if 'id' in item:
			id = int(item.split(':')[1])
			e.append(id)
		if 'mode' in item:	
			e.append(entry[i+1])
		if 'time' in item:
			t1 = entry[i+1].split('-')[0]
			t2 = entry[i+1].split('-')[1]
			e.append(t1)
			e.append(t2)			
	return e	
	

def decodeEntry(name, entry):
	
	decoded = []
	
	if types_dict[name]=="symbolic":
		decoded = decodeSymbolic(entry)
	elif types_dict[name]=="numeric":
		decoded = decodeNumeric(entry)		
	elif types_dict[name]=="bucket":
		decoded = decodeBucket(entry)		
	elif types_dict[name]=="wind":
		decoded =  decodeWind(entry)		

	return decoded
	
		
def decodeNumeric(entry):
	
	elements = ['.id', '#min', '#max', '#mean']
	new_entry = []
	
	for i, item in enumerate(entry):
		new_entry.append(elements[i])
		new_entry.append(item)		
		
	return new_entry	
		
				
	
def decodeSymbolic(entry):

	elements = ['.id', '@mode']
	new_entry = []
	
	for i, item in enumerate(entry):
		new_entry.append(elements[i])
		new_entry.append(item)	
	return new_entry
		

def decodeWind(entry):
	
	elements = ['.id', '@mode']
	new_entry = []
	
	for i, item in enumerate(entry):
		new_entry.append(elements[i])
		new_entry.append(item)	
	return new_entry			


def decodeBucket(entry):
	
	elements = ['.id', '#min', '#max', '#mean', '@mode-bucket (range)', '-']
	new_entry = []
	
	for i, item in enumerate(entry):
		new_entry.append(elements[i])
		new_entry.append(item)	
	return new_entry	
		
		
names_dict = {
'.id:0' : 'temperature', 
'.id:1' : 'windChill', 
'.id:2' : 'windSpeed', 
'.id:3' : 'windDir', 
'.id:4' : 'gust', 
'.id:5' : 'skyCover', 
'.id:10' : 'precipPotential', 
'.id:11' : 'thunderChance', 
'.id:12' : 'thunderChance', 
'.id:13' : 'thunderChance', 
'.id:14' : 'thunderChance', 
'.id:15' : 'thunderChance', 
'.id:16' : 'rainChance', 
'.id:17' : 'rainChance', 
'.id:18' : 'rainChance', 
'.id:19' : 'rainChance', 
'.id:20' : 'rainChance', 
'.id:21' : 'snowChance', 
'.id:22' : 'snowChance', 
'.id:23' : 'snowChance', 
'.id:24' : 'snowChance', 
'.id:25' : 'snowChance'}

types_dict = {
"temperature" : "numeric",
"windChill" : "numeric",
"windSpeed" : "bucket",
"windDir" : "wind",
"gust" : "numeric",
"skyCover" : "bucket",
"precipPotential" : "numeric",
"thunderChance" : "symbolic",
"rainChance" : "symbolic",
"snowChance" : "symbolic",
"freezingRainChance" : "symbolic",
"sleetChance" : "symbolic",

}

### Parameters and utility methods above
### Main programme flows starts here.

input = "./data/weather-events2-new.txt"
#target = "./data/alabama/alabaster/2009-02-07-0.text"
file_dir_stage1 = "./data/data_stage1/"
file_dir_stage2 = "./data/data_stage2/"


weather_records = {}
states = []
dates = []
times = []
state_cities = {}
current_state = "alabama"
cities = []

for line in open(input, 'r'):
	f = line.split("===")[0]
	weather_records[f] = line
	f = f.split('.text')[0]	
	state = f.split('/')[2]
	city = f.split('/')[3]	
	if not current_state == state:
		state_cities[current_state] = cities
		cities = []		
		current_state = state
	else:
		if not city in cities:
			cities.append(city)	
	date1 = f.split('/')[4]
	date = '-'.join(date1.split('-')[:-1])
	time = date1.split('-')[-1]
	if not state in states:
		states.append(state)
#	if not city in cities:
#		cities.append(city)		
	if not date in dates:
		dates.append(date)
	if not time in times:
		times.append(time)		
			

data = []
data_tuples = []
models_dict = {}
data_dict = {}
models = {}
forecast_tuples = []
forecast_models = {}
forecast_models_dict = {}
forecast_data_dict = {}

		
# Iterate through the data to gather the model parameters we need know to load pre-trained
# models and make predictions with them. This is for STAGE 1.
 
# Go through all the weather files and prepare data sets. 
for file in os.listdir(file_dir_stage1):
    if file.endswith(".txt"):
    	modelName = file.split('.txt')[0]
    	print modelName
    	X_set = []
    	Y_set = []    	
    	for line in open(file_dir_stage1+file):
    		l = line.replace('\n', '')
    		X_set.append(l.split('===')[0].split())
    		label = int(l.split('===')[1])
    		Y_set.append(label)
    models[modelName] = (X_set, Y_set)

# Iterate through all task-specific models (wind, rain, temperature etc) and determine their characteristics.
# This is for STAGE 1 (predicting inclusion of weather events).

for key in models:	
	print '***'*30
	print 'Designing model:' + colors.highlight+ ' ***' + key + '***' + colors.close
	print '(DataSet)'

	sub_dataset = DataSet(np.asarray(models[key][0]), np.asarray(models[key][1]))
	sub_dataset.vocab = sub_dataset.countSymbols()
	sub_dataset.max_len = sub_dataset.countMaxLen()
	X, Y = sub_dataset.representData(models[key][0], models[key][1])
	X_train, X_test, Y_train, Y_test = sub_dataset.shuffleAndSplit(X, Y)
	data_dict[key] = sub_dataset
	print '-'*50		

	# Initiate a DL based on data specifications, then load the weights we previously trained.
	print '(DeepLearner)'	
	sub_deep = DeepLearner('LSTM', layers=2, hidden_size=20, epochs=10)	
	model = sub_deep.designModel(sub_dataset.max_len, sub_dataset.vocab, sub_dataset.dimensionality)
	model_file = key+'-weights.h5'
	sub_deep.model.load_weights("./data/out_stage1/"+model_file)
	sub_deep.compileModel(model)	
	models_dict[key] = sub_deep.model
		

# Iterate through all task-specific forecast models (wind, rain, temperature etc) and determine their characteristics.
# This is for STAGE 3 (predicting forecasts).

# Go through all the weather files and prepare data sets. 
for file in os.listdir(file_dir_stage2):
    if file.endswith(".txt"):
    	modelName = file.split('.txt')[0]
    	X_set = []
    	Y_set = []    	
    	for line in open(file_dir_stage2+file):
    		l = line.replace('\n', '')
    		X_set.append(l.split('===')[0].split())
    		Y_set.append(l.split('===')[1].split())    			
    forecast_models[modelName] = (X_set, Y_set)

# Now iterate through all the models, and determine the best input-output representation.
for key in forecast_models:	
	print '***'*30
	print 'Designing model:' + colors.highlight+ ' ***' + key + '***' + colors.close
	print '(DataSet)'

	# analyse data and represent as needed, count some items we need to know.
	sub_dataset = DataSet(np.asarray(forecast_models[key][0]), np.asarray(forecast_models[key][1]))
	sub_dataset.vocab = sub_dataset.countSymbols()
	sub_dataset.max_len = sub_dataset.countMaxLen()
	X, Y = sub_dataset.representData(forecast_models[key][0], forecast_models[key][1])
	X_train, X_test, Y_train, Y_test = sub_dataset.shuffleAndSplit(X, Y)
	forecast_data_dict[key] = sub_dataset	
	print '-'*50		

	# now choose and compile a deep learning model, train it and save the outputs.
	print '(DeepLearner)'	
	sub_deep = DeepLearner('LSTM', layers=4, hidden_size=20, epochs=2000)
	model = sub_deep.designModel(sub_dataset.max_len, sub_dataset.vocab, sub_dataset.dimensionality)
	model_file = key+'-weights.h5'
	sub_deep.model.load_weights("./data/out_stage2/"+model_file)
	sub_deep.compileModel(model)	
	forecast_models_dict[key] = sub_deep.model	
	
print '-'*100
print colors.bold + "All loaded and ready..." + colors.close 
print '-'*100	
	
# Ask user to choose a place and time to get weather for.
user_state = raw_input("Please choose a state: " + str(states[:-1]) + "\n")
user_city = raw_input("Please choose a city: " + str(state_cities[user_state]) + "\n")
user_date = raw_input("Please choose a date: " + str(dates) + "\n")
user_time = raw_input("Please choose a time [0=morning, 1=evening]: " + str(times) + "\n")

weather_for = "./data/"+user_state+"/"+user_city+"/"+user_date+"-"+user_time+".text"

print "Finding weather for: " + weather_for
target = weather_for

# Find the correct data record and represent data in a way understood by DL agent. 
# Store newly represented data as tuples for later access. 

for line in open(input, 'r'):
	if line.split('===')[0] == target:
		data = line.split("===")
included = data[2]	
ordered = data[3]
forecast = data[4].replace('\n', '').split('|')	

# Stage 1 - iterate through the events in the record, re-represent the ones to include.
# Store the newly represented events in tuples alongside their expected forecasts.

for d in data[1].split("|"):
	if '.type' in d:
		type = str(d.split()[2])
		vector = []
		output = 0
		if d.split()[0] in included:
			output = 1
		if types_dict[type] == 'numeric':
			vector = getNumeric(d.split())
		elif types_dict[type] == 'symbolic':	
			vector = getSymbolic(d.split())		
		elif types_dict[type] == 'bucket':	
			vector = getNumericBucket(d.split())					
		elif types_dict[type] == 'wind':	
			vector = getSymbolicWind(d.split())							
		data_tuples.append((type, vector, output))	
print data_tuples
	
# Moving to Stage 3 - iterate through the events in the record, re-represent the ones to include.
# Store the newly represented events in tuples alongside their expected forecasts.

ordered = ordered.split("|")
for o, otem in enumerate(ordered):
	if not 'id' in otem:
		del ordered[o]
		del forecast[o]			
if not len(ordered)==len(forecast):
	print 'problem'

template = list(range(0, len(ordered)))
for p, ptem in enumerate(ordered): 
	item = ptem.strip().split()
	record = ''
	for i in item:
		for e in data[1].split("|"):
			if 'id' in e:
#				print e.split()
				if e.split()[0] == i:
					e0_type = types_dict[e.split()[2]]
					if e0_type == "numeric":
						e1 = str(getNumericS3(e.split()))
					elif e0_type == "symbolic":	
						e1 = str(getSymbolicS3(e.split()))
					elif e0_type == "bucket":	
						e1 = str(getNumericBucketS3(e.split()))
					elif e0_type == "wind":	
						e1 = str(getSymbolicWindS3(e.split()))
					if e1 == '':
						e1 = e
						print 'e', e
					record = record + e1 + ' '
	template[p] = record.strip()
new_record = ''	
for t, ttem in enumerate(template):
	ttem1 = ttem.replace('[', '').replace(']', '').split()
	template[t] = ttem1
for q, qtem in enumerate(ordered):
	for x1, xtem1 in enumerate(qtem.split()):
		xt1 = names_dict[xtem1]
		qtem = qtem.replace(xtem1, xt1)			
	m_name = qtem.strip().replace(' ', '_').replace('.', '').replace(':', '')
	m_vector = str(template[q]).replace(",", '').replace("'", "").replace('"', '').replace('[', '').replace(']', '').split()
	m_output = forecast[q]
	forecast_tuples.append((m_name, m_vector, m_output))	

print '-'*100
print colors.bold + "Now making predictions for " + colors.close + target + " ..."
print colors.bold + "STAGE 1: Predicting the inclusion of weather events in forecast. (1 = included, 0 = not included)" + colors.close
print '-'*100
	
for e in data_tuples:
	name = e[0]
	entry = e[1]
	entry_encoded = e[1][:]
	output = e[2]	
	learner = models_dict[name]
	dataset = data_dict[name]
	for j, jtem in enumerate(entry_encoded):
		entry_encoded[j] = dataset.char_indices[str(jtem)]
	
	pred = learner.predict_classes(np.asarray([entry_encoded]), verbose=0)
	entry_decoded = decodeEntry(name, entry)
	entry_decoded = str(entry_decoded).replace(',', '').replace("'", '')
	if str(output) in str(pred):
		if '1' in str(pred):
			c = colors.ok + "(correct)" + colors.close
		else:	
			c =  "(correct)" 
		print colors.highlight + name + colors.close + " " + str(entry_decoded) + " --> " + "Prediction:" + " " + str(pred) + " " +c			
	else:
		print colors.highlight + name + colors.close + " " + str(entry_decoded) + " --> " + "Prediction:" + " " + str(pred) + colors.fail + " " + "(incorrect)" + colors.close			
	
print '-'*100

print '-'*100
print colors.bold + "Including events (in descending order of priority)" + colors.close + str(included) + " ... (implement me)"
print colors.bold + "STAGE 2: Aggregate thematically similar events: " + colors.close + str(ordered) + ". " + "(implement me)"
print '-'*100


print '-'*100
print colors.bold + "STAGE 3: Generating weather forecast for... " + colors.close + " " +target
print '-'*100
	
expected_forecast = []
predicted_forecast = []	
	
for e in forecast_tuples:
	name = e[0]
	entry = e[1]
	entry_encoded = e[1][:]
	output = e[2]	
	expected_forecast.append(e[2])
	learner = forecast_models_dict[name]
	dataset = forecast_data_dict[name]

	X1 = np.zeros((dataset.max_len, len(dataset.vocab)))
	for j, c in enumerate(entry_encoded):
		X1[j, dataset.char_indices[c]] = 1
	entry_encoded = X1
	
	preds = learner.predict_classes(np.asarray([entry_encoded]), verbose=0)
	sent = ' '.join(dataset.indices_char[p] for p in preds[0])
	if ',' in sent:
		sent = sent.split(",")[0]+", "
	if '.' in sent:
		sent = sent.split(".")[0]+". "
	predicted_forecast.append(sent)
	
print colors.highlight + "Expected forecast... " + colors.close	
print ' '.join(expected_forecast) + '\n'
	
print colors.highlight + "Predicted forecast..." + colors.close
print ' '.join(predicted_forecast)

print '-'*100
