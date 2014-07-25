import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
from sklearn import cross_validation
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import numpy
import time


def targetMapper(str):
    if str == u'1':
        return 1
    return 0

def sampleTextFields(rows, sampleSize, vec, target, numericFeats, ids, hasTarget=True):
    sample = random.sample(xrange(len(rows)), sampleSize)
    for index in sample:
        row = rows[index]
        text = row['title'] + " " + row['description'] + " " + row['category'] + " " + row['subcategory'] + " " + row['attrs']
        vec.append(text)
        ids.append(row["itemid"]);

        arr = []
        arr.append(row['price'])
        arr.append(row['phones_cnt'])
        arr.append(row['emails_cnt'])
        arr.append(row['urls_cnt'])
        #titleLength = len(row['title'].split())
        #textLength = len(text.split())
        #arr.append(titleLength)
        #arr.append(textLength)
        numericFeats.append(arr)

        if hasTarget == True:
            target.append(targetMapper(row['is_blocked']))

def loadPickledFile(fileName):
    fl = open(fileName, 'rb')
    pcl = pickle.load(fl)
    fl.close()
    return pcl;

if __name__=="__main__":
    t0 = time.time()
    blocked = loadPickledFile('output/trainblocked.pkl')
    t1 = time.time()
    print "unpicked blocked, took ",(t1-t0)

    t0 = time.time()
    unblocked = loadPickledFile('output/trainunblocked.pkl')
    t1 = time.time()
    print "unpicked unblocked, took ",(t1-t0)

    sampleSize = len(blocked)
    vec = []
    target = []
    docIds = []
    numerifFeats = []

    t0 = time.time()
    sampleTextFields(blocked, sampleSize, vec, target, numerifFeats, docIds)
    sampleTextFields(unblocked, sampleSize, vec, target, numerifFeats, docIds)
    t1 = time.time()
    print "Done sampling, took ", (t1-t0), " Sample size: ", len(vec)

    del blocked
    del unblocked

    t0 = time.time()
    wordVectorizer = TfidfVectorizer(ngram_range=(1,1), analyzer="word", binary=False, min_df=2, smooth_idf=True, sublinear_tf=True, use_idf=True, max_features=50000, max_df=0.8, lowercase=True)
    trainTfIdf = wordVectorizer.fit_transform(vec)
    #charVectorizer = TfidfVectorizer(ngram_range=(1,5), analyzer="char", binary=False, min_df=2, smooth_idf=True, sublinear_tf=True, use_idf=True, max_features=50000, max_df=0.8, lowercase=True)
    #charTrainTfIdf = charVectorizer.fit_transform(vec)
    t1 = time.time()
    print "vectorized. Took ", (t1-t0)


    t1 = time.time()
    print "Done learning, ", (t1-t0)
    #print scores

    #### Testing
    testing = True
    if testing:
        testVec = []
        testIds = []
        testTargets = []
        numericFeatures = []


        clf = lm.LogisticRegression(penalty='l1', dual=False, tol=1e-06, C=3, fit_intercept=True,intercept_scaling=0.1)
        clf = clf.fit(trainTfIdf, numpy.array(target))

        t0 = time.time()
        testData = loadPickledFile('output/testUnblocked.pkl')
        t1 = time.time()
        print "unpicked test file, took ",(t1-t0)

        sampleTextFields(testData, len(testData), testVec, testTargets, numericFeatures, testIds, False)
        testTfIdfs = wordVectorizer.transform(testVec)
        predics_probas = clf.predict_proba(testTfIdfs)

        zippedProbaWeight = sorted(zip(predics_probas[:,1], testIds), reverse=True)

        f = open('output/predict2.txt','w')
        for entry in zippedProbaWeight:
            line = str(entry[1]) + "\n"
            f.write(line)
        f.close()
    else:
        t0 = time.time()

        target = numpy.array(target)
        numerifFeats = numpy.array(numerifFeats, dtype=int)

#        #paramter grid search
#        tunedParams = [{
#            'penalty': ['l1'],
#            'tol': [1e-5, 1e-8, 1e-10],
#            'C': [1, 2, 3, 4],
#            #'fit_intercept': [True, False]
#            }]
#
#        X_train, X_test, y_train, y_test = train_test_split(numerifFeats, target, test_size=0.10, random_state=0)
#
#        clf = GridSearchCV(GradientBoostingClassifier(verbose=1), tuned_parameters, scoring='precision', cv=3, n_jobs=1, verbose=10)
#        #clf = GridSearchCV(lm.LogisticRegression(dual=False, intercept_scaling=1.0), tunedParams, cv=3, scoring='precision', n_jobs=2, verbose=5)
#        clf.fit(X_train, y_train)
#        print "Best parameters set found on development set:"
#        print ""
#        print clf.best_estimator_
#        print "Grid scores on development set:"
#
#        for params, mean_score, scores in clf.grid_scores_:
#            print("%0.3f (+/-%0.03f) for %r"
#            % (mean_score, scores.std() / 2, params))
#            print ""
#        print "Detailed classification report:"
#        print ""
#        print "The model is trained on the full development set."
#        print "The scores are computed on the full evaluation set."
#        print ""
#        y_true, y_pred = y_test, clf.predict(X_test)
#        print classification_report(y_true, y_pred)
#        print ""
#

        scores = []
        # split into train and test
        rs = cross_validation.ShuffleSplit(len(target), 5, test_size=0.2)
        for trainIndex, testIndex in rs:

            clf = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0000001, C=3, fit_intercept=True,intercept_scaling=0.1)
            clfNum = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0000001, C=3, fit_intercept=True,intercept_scaling=1.0)

            wordPredict = clf.fit(trainTfIdf[trainIndex,:], target[trainIndex]).predict_proba(trainTfIdf[testIndex,:])[:,1]
            numericFeatsPredict = clfNum.fit(numerifFeats[trainIndex,:], target[trainIndex]).predict_proba(numerifFeats[testIndex,:])[:,1]

            #stacked = zip(wordPredict, numericFeatsPredict)
            stacked = zip(wordPredict, wordPredict)
            sumed = [round((entry[0]*0.8 + entry[1]*0.2),0) for entry in stacked]
            score = precision_score(target[testIndex], sumed)
            scores.append(score)
            print "Learned ...", score


        print scores
