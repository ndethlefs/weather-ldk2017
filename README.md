# Code for "Deep Text Generation -- Using Hierarchical Decomposition to Mitigate the Effect of Rare Data Points"

This repository contains code for the following paper:

N Dethlefs and A Turner (2017) Deep Text Generation -- Using Hierarchical Decomposition to Mitigate the Effect of Rare Data Points. International Conference on Language, Data and Knowledge, pp. 290-298. Dublin, Ireland.


Please link here: https://link.springer.com/chapter/10.1007/978-3-319-59888-8_25


The basic idea of this paper is that an application domain in natural language generation -- that is otherwise large 
and difficult to learn from due to data sparcity and imbalance -- can be broken up into a number of organised "sub-domains" 
to ease training. As a mitigation to the problem of unbalanced training data, we propose to decompose a large natural language dataset into 
several subsets that “talk about” the same thing. We show that the decomposition helps to focus each learner’s attention during 
training. Results from a proof-of-concept study show 73% times faster learning over a flat model and better results.

In the paper, we work with a hand-crafted hierarchy looking separately into traditional NLG tasks content selection, 
microplanning and surface realisation.

<img src="/img/hierarchy.png" alt="drawing" width="700"/>


We further decompose domain-specific tasks (in this paper we're using weather forecast generation as an example), e.g. 
into wind, rain, clouds, etc.

<img src="/img/weather-hierarchy.png" alt="drawing" width="700"/>

All hierarchies are hand-crafted in this paper (or obtained via heuristics). This process could be automated. Please see https://github.com/ndethlefs/kgen2 for an example of this in a different domain.

# Code and Data

We're using the dataset of Liang et al 2009: http://cs.stanford.edu/~pliang/papers/weather-data.zip

The code comes with four scripts:

<code>stage1-content.py</code> trains the content selection component to decide which weather events to mention and which not. 

<code>stage2-sequence.py</code> orders the chosen events (from stage 1) into the sequence in which they should be presented.

<code>stage3-forecast.py</code> uses the output of stage 2 and generates a text-based weather forecast.

These could be combined into the same process but were designed separately here.

Finally, to test the trained models:

<code>test_weather_output.py</code> uses pre-trained weights (that get saved during training) 
and output files to generate forecasts for a small number of states. Again, this could be extended but was kept
as a small demonstration here due to the time it takes to load in all the data.


