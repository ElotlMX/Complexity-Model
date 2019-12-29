from re import sub
from collections import defaultdict
from itertools import chain
import numpy as np

class nPhones:
	#file <- Archivo de entrada
	#nphone_siz <- Tamaño de los n-phones/grams que se obtendrán. Si no hay argumento, n=3
	def __init__(self,file,nphone_siz=3):
		super().__init__()
		#Se eliminan signos del archivo de entrada
		self.file = sub(r'[^\w\s]','',file.strip().lower()).split()
		self.ngram_siz = nphone_siz

	#Extractror de ngramas
	def ngramas(self,string):
		#Se agrega el simbolo de inicio '#' y el de final '$'
		string = '#'+string+'$'
		ngrams = []
		#Se cehca la longitud de la cadena de entrada
		if len(string) < self.ngram_siz:
			ngrams.append(string)
		#Se extraen los ngramas
		else:
			i = 0
			while i + self.ngram_siz - 1 < len(string):
				ngrams.append(string[i:i + self.ngram_siz])
				i += 1
		#Return una lista de ngramas
		return(ngrams)

	#Se obtienen nphones (ngramas a nivel palabra)
	def n_phones(self):
		word_phones = []
		for w in self.file:
			word_phones.append(self.ngramas(w))
		#Regresa una lista de los nphones de cada palabra
		return(word_phones)

	#Define un vocabulario que indexa los nphones {idx:nphone}
	def vocab(self):
		vocab = defaultdict()
		vocab.default_factory = lambda: len(vocab)
		return(vocab)

	#Sustituye los nphones por sus índices númericos
	def word_idx(self,corpus, vocab):
		for doc in corpus:
			yield([vocab[w] for w in doc])

	#Cadena de nphones
	word_phones = None
	#Cadenas de índices numéricos de nphones
	idx_phones = None
	#Vocabulario índice:nphone
	voc = None

	#Obtiene los nphones, el vocabulario, y las cadenas con índices
	def get_phones(self):
		self.word_phones = self.n_phones()
		self.voc = self.vocab()
		self.idx_phones = list(self.word_idx(self.word_phones, self.voc))



