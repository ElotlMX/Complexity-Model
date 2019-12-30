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

