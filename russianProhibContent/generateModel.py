import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
from sklearn import cross_validation
#from sklearn.ensemble import RandomForestClassifier
import numpy
import time


def targetMapper(str):
	if str == u'1':
		return 1
	return 0

def sampleTextFields(rows, sampleSize, vec, target, ids, hasTarget=True):
	sample = random.sample(xrange(len(rows)), sampleSize)
	for index in sample:
		row = rows[index]
		text = row['title'] + " " + row['description'] + " " + row['category'] + " " + row['subcategory']
		vec.append(text)
		ids.append(row["itemid"]);
		if hasTarget == True:
			target.append(targetMapper(row['is_blocked']))
		
def loadPickledFile(fileName):
	fl = open(fileName, 'rb')
	pcl = pickle.load(fl)
	fl.close()
	return pcl;

if __name__=="__main__":
	t0 = time.time()
	blocked = loadPickledFile('output/blocked.pkl')
	t1 = time.time()
	print "unpicked blocked, took ",(t1-t0)
 
	t0 = time.time()
	unblocked = loadPickledFile('output/unblocked.pkl')
	t1 = time.time()
	print "unpicked unblocked, took ",(t1-t0)
	
	sampleSize = len(blocked)
	vec = []
	target = []
	docIds = []

	t0 = time.time()
	sampleTextFields(blocked, sampleSize, vec, target, docIds)
	sampleTextFields(unblocked, sampleSize, vec, target, docIds)
	t1 = time.time()
	print "Done sampling, took ", (t1-t0), " Sample size: ", len(vec)

	t0 = time.time()
	countvect_word = TfidfVectorizer(ngram_range=(1,1), analyzer="word", binary=False, min_df=2, smooth_idf=True, sublinear_tf=True, use_idf=True, max_features=50000, max_df=0.8)
	trainTfIdf = countvect_word.fit_transform(vec)
	t1 = time.time()
	print "vectorized. Took ", (t1-t0)

	t0 = time.time()	
	clf = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True,intercept_scaling=1.0)
	clf = clf.fit(trainTfIdf, numpy.array(target))
	#clf = RandomForestClassifier(n_estimators=1000)
	#clf = clf.fit()
	t1 = time.time()
	print "Done learning, ", (t1-t0)
	#print scores

	#### Testing
	testing = False
	if testing:
		testVec = []
		testIds = []
		testTargets = []
		
		t0 = time.time()
		testData = loadPickledFile('output/testUnblocked.pkl')
		t1 = time.time()
		print "unpicked test file, took ",(t1-t0)
	
		sampleTextFields(testData, len(testData), testVec, testTargets, testIds, False)
		testTfIdfs = countvect_word.transform(testVec)
		predics_probas = clf.predict_proba(testTfIdfs)
		
		zippedProbaWeight = sorted(zip(predics_probas[:,1], testIds), reverse=True)
		
		f = open('output/predict1.txt','w')
		for entry in zippedProbaWeight:
			line = str(entry[1]) + "\n"
			f.write(line)
		f.close()
	else:
		scores = cross_validation.cross_val_score(clf, trainTfIdf, numpy.array(target), cv=5, scoring='precision')
		print scores
