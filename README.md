# Productivity and Predictability for Measuring Morphological Complexity


Repository that contains the code for calculating the entropy rate of a subword language model. This is part of the article [Productivity and Predictability for Measuring Morphological Complexity](https://www.mdpi.com/1099-4300/22/1/48)

This program runs in python 3. The program uses the next libraries:

* Standard pyhton libraries (numpy, collections, itertools, random, re)
* nltk (Natural Language Toolkit) https://www.nltk.org/ 

## Basic Usage

``python main.py --input_directory``

input_directory should be a directory containing a parallel corpus, where each file corresponds to a language (each file must be already tokenized). 

Corpora for the languages mentioned in the article were pre-processed and extracted from:
- [The Parallel Bible Corpus](http://www.christianbentz.de/MLC2019_data.html) 
- [JW300](http://opus.nlpl.eu/JW300.php)

### Parameters for the entropy rate of the neural probabilistic language model:
* n : the size of n-grams. Default is 3
* iter : number of iterations to train the neural network. Default is 50
* emb_dim : Number of dimensions in embedding vectors. Default is 300
* hid_dim : Number of dimensions in hidden layer. Default is 100

To run the model with different parameters, execute the program as in the following example:

``python3 main.py --input_directory --output results/results.csv --n 1 --iter 100``



