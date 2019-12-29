# coding: utf-8
import numpy as np
from collections import defaultdict, Counter
from itertools import chain
from nltk.util import ngrams
from random import sample

#Funcion que crea un vocabulario de palabras con un indice numerico
def vocab():
    vocab = defaultdict()
    vocab.default_factory = lambda: len(vocab)
    return vocab    

#Funcion que pasa la cadena de simbolos a una secuencia con indices numericos
def text2numba(corpus, vocab):
	for doc in corpus:
		yield [vocab[w] for w in doc]
	
class Model():
	def __init__(self,strings,dim = 300,nn_hdim=100,ngramas=2):
		super().__init__()
		#Se obtiene el voc de indices y las cadenas indexadas
		self.voc = vocab()
		self.cadenas = list(text2numba(strings,self.voc))
		self.size = ngramas
		self.N = len(self.voc)
		np.random.seed(0)
		#Embedding
		self.emb = np.random.randn(self.N, dim) / np.sqrt(self.N)
		#Capa oculta
		self.U = np.random.randn(dim, nn_hdim) / np.sqrt(dim)
		self.b = np.zeros((1, nn_hdim))

		#Capa de salida
		self.W = np.random.randn(nn_hdim,  self.N) / np.sqrt(nn_hdim)
		self.c = np.zeros((1, self.N))

	def train(self, its=100,eta=0.1, batch=100):
	# Se entrena el modelo. Para esto, se define un número de iteraciones y un rango de aprendizaje. 
	#Se utiliza backpropagation y gradient descent.
		examples = list(chain(*[ngrams(cad,self.size) for cad in self.cadenas]))
		
		#Tomara solo un batch de ejemplos
		for t in range(0,its):
			err = 0.0
			for ex in sample(population=examples, k=batch):
				#Forward
				#Embedding
				c_w = (1/(self.size-1))*np.sum(self.emb[ex[i]] for i in range(0,self.size-1))
				#capa oculta
				h1 = np.tanh(np.dot(self.U.T,c_w) + self.b)[0]
				#salida
				out = np.exp(np.dot(self.W.T,h1) + self.c)[0]
				#Softmax
				f = out/out.sum(0)

				#Backprop
				#Variable de salida
				d_out = f
				d_out[ex[-1]] -= 1
				
				#error
				err += (f*np.log(f+1)).sum(0)				

				#Variable para la capa oculta
				d_tanh = (1-h1**2)*np.dot(self.W,d_out)
				
				#Variable de embedding
				d_emb = np.dot(self.U,d_tanh)
			
				#Actualizacion de salida
				for j in range(len(d_out)):
					self.W.T[j] -= eta*d_out[j]*h1

				#Actualiza bias de salida
				self.c -= eta*d_out[j]

				#Actualizacion de capa oculta
				for j in range(len(d_tanh)):
					self.U.T[j] -= (eta*d_tanh[j]*c_w)

				#Actualiza bias
				self.b -= eta*d_tanh

				#Actualizacion de embedding
				self.emb[ex[0]] -= eta*(1/(self.size-1))*d_emb	
				

			if t%10 == 0: print('Log error in iteration', t, 'is', err/self.N)

	#Forward
	def forward(self,x):    
		#Embedimiento
		c_w = (1/(self.size-1))*np.sum(self.emb[self.voc[x[i]]] for i in range(0,self.size-1))
		#print c_w.shape, U.T.shape
		h1 = np.tanh(np.dot(self.U.T,c_w) + self.b)[0]
		#print h1 #W.shape, h1.shape
		out = np.exp(np.dot(self.W.T,h1) + self.c)[0]
		#print out
		p = out/out.sum(0)
		return p
