Code for our LDK 2017 paper:

N Dethlefs and A Turner (2017) Deep Text Generation -- Using Hierarchical Decomposition to Mitigate the Effect of Rare Data Points. International Conference on Language, Data and Knowledge, pp. 290-298. Dublin, Ireland.

stage1-content.py trains the content selection component to decide which weather events to mention and which not. 

stage2-sequence.py orders the chosen events (from stage 1) into the sequence in which they should be presented.

stage3-forecast.py uses the output of stage 2 and generates a text-based weather forecast.

Using the dataset of Liang et al 2009: http://cs.stanford.edu/~pliang/papers/weather-data.zip

test_weather_output.py uses pre-trained weights and output files to generate forecasts for a small number of states (because it takes long to load).