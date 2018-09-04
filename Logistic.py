import re
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

#from gensim.test.utils import common_texts
#from gensim.sklearn_api import W2VTransformer


def remove_ref(ss):
	'''
	Does some basic text processing. Like the header and the non-alpha
	numerics.
	'''
	for i in range(len(ss)):
		s = ss[i]
		ref_str = re.search(r'==(.*)==',s)
		if ref_str is not None:
			s = s.replace(ref_str.group(),"")
		s = s.replace("...",".")
		s = s.replace("..",".")
		#print(ss[i])
		
		f = lambda w: "".join([char.lower() for char in w 
								if char.isalpha()])
		w_list = [w for w in map(f, s.split()) if w is not ""]
		s = " ".join(w_list)
		ss[i] = s
	return ss


def cal_accuracy(true, pred):
	'''
	Calculates the accuracy of the prediction
	'''
	total = 0.0
	acc = 0.0
	for i in range(true.shape[0]):
		for j in range(pred.shape[1]):
			if true[i,j]!=-1:
				if true[i,j]==pred[i,j]:
					acc+=1
				total+=1
	print(acc/total)


def add_extra_label(Y):
	'''
	Adds an extra label for 'normal' comments
	'''
	_Y = np.zeros((Y.shape[0],Y.shape[1]+1))
	_Y[:,:-1]=Y
	for i in range(0,_Y.shape[0]):
		c=0
		for j in range(0,_Y.shape[1]):
			if _Y[i,j]==1:
				c=1
		if c==0:
			_Y[i,-1]=1 
	return _Y


def train_model(idx):
	'''
	Trains a model per class (binary classification) and generates
	the prediction for that class
	'''
	ones = np.sum(trainY[:,idx], axis=0)
	tots = trainX.shape[0]
	print("Ones=%d, Zeros=%d"%(ones,tots))
	model = LogisticRegression(class_weight="balanced")
	model.fit(trainX, trainY[:,idx])
	return model.predict(testX)


#def trim_training_data(X,y, tol=0.4):
	'''
	Manual class balancing. Not used currently. Uses weighted
	update methods at the moment.
	'''
#	#for i in range(X.shape[0]):
#	ones = np.sum(Y, axis=0)
#	tots = X.shape[0]
#	if ones < ceil(0.4*ones):
	
#	print(_sum_Y)
		
#	return X,Y


if __name__ == "__main__":

	# Prepare Train Data
	df_train = pd.read_csv("train.csv")
	print(df_train.head())
	#print(df_train.dtypes)
	trainY = np.array(df_train.loc[:,"toxic":])
	
	s_list_train = list(df_train["comment_text"][:])
	s_list_train = remove_ref(s_list_train)
	
	#trim_training_data(s_list_train, trainY)
	#trainY = add_extra_label(trainY)
	
	# Prepare Test Data
	df_test = pd.read_csv("test.csv")
	print(df_test.head())
	s_list_test = list(df_test["comment_text"][:])
	s_list_test = remove_ref(s_list_test)

	df_op = pd.DataFrame({'id':df_test['id']})
	#testY = add_extra_label(testY)
	#df_op[:,"toxic":]= testY
	
	# Create BOW Model
	vectorizer = CountVectorizer(stop_words="english")
	vectorizer.fit(s_list_train)

	trainX = vectorizer.transform(s_list_train)
	testX = vectorizer.transform(s_list_test)
	
	# Train and Test (One model for each class)	
	predY=np.zeros((testX.shape[0],6),dtype='float')
	for i in range(6):
		predY[:,i]=train_model(i)
		
	label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat',\
					'insult', 'identity_hate']
	df_op = pd.concat([df_op, pd.DataFrame(predY, columns=label_cols)],axis=1)
	df_op.to_csv("op.csv",index=False)
	
