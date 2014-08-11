import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import sklearn.linear_model as lm
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier

import numpy
import time
from sklearn.preprocessing import OneHotEncoder


def targetMapper(str, isProvedIlicit=u'0'):
    if str == u'1':
        if isProvedIlicit == u'1':
            return 2
        return 1
    return 0

def cleanUpData(rows, sampleSize, features, hasTarget=True):

    countofApproved = {}
    durationOfAdd = []

    sample = random.sample(xrange(len(rows)), sampleSize)
    for index in sample:
        row = rows[index]

        #ad id
        features["ids"].append(row["itemid"])

        ###### textual work
        # concat attribute keys and valus into a single string
        attrs = ''
        for keyz in row['attrs']:
            attrs += " " + keyz + " " + row['attrs'][keyz]
        attrs = attrs.strip()

        # textual parts will be the title description and attributes
        #
        text = row['title'] + " " + row['description'] + " " + attrs + " "
        features["textDescription"].append(text)


        ###### nuermic features
        arr = []

        # does the list contain a phone #?
        hasPhone = 0
        if (int(row['phones_cnt']) > 0):
            hasPhone = 1
        arr.append(hasPhone)

        # boolean if it has an email
        hasEmail = 0
        if (int(row['emails_cnt']) > 0):
            hasEmail = 1
        arr.append(hasEmail)

        # boolean attribute if it has a URL
        hasUrls = 0
        if (int(row['urls_cnt']) > 0):
            hasUrls = 1
        arr.append(hasUrls)

        # encode category and sub category
        entry = row['subcategory']
        if entry not in features["lookupSubCats"]:
            features["lookupSubCats"][entry] = len(features["lookupSubCats"])
        arr.append(features["lookupSubCats"][entry])

        entry = row['category']
        if entry not in features["lookupCats"]:
            features["lookupCats"][entry] = len(features["lookupCats"])
        arr.append(features["lookupCats"][entry])


        arr.append(float(row['price']))

        # additional feature count words in title
        titleLength = len(row['title'].split())
        arr.append(titleLength)

        # count of words in the description
        textLength = len(text.split())
        arr.append(textLength)

        # depth of the attributes
        arr.append(len(row['attrs']))



        if hasTarget == True:
            features["target"].append(targetMapper(row['is_blocked']))

        features["numericDescription"].append(arr)

        if row['is_proved'] not in countofApproved:
            countofApproved[row['is_proved']] = 0
        countofApproved[row['is_proved']] += 1

        if row['close_hours'].strip() != '':
            durationOfAdd.append(float(row['close_hours'])*60)




    print countofApproved
    durationOfAdd = numpy.array(durationOfAdd)
    print "duration mean: ", numpy.mean(durationOfAdd)
    print "duration stddev: ", numpy.std(durationOfAdd)
    print "duration median: ", numpy.median(durationOfAdd)


def encodeCategoricalStringFeats(rows):
    lookUp = {}
    arr = []
    for entry in rows:
        if entry not in lookUp:
            lookUp[entry] = len(lookUp)
        arr.append(lookUp[entry])
    return numpy.array(arr)


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
    rows = {}

    rows["textDescription"] = [];
    rows["numericDescription"] = [];
    rows["categoryDescription"] = [];
    rows["subcategoryDescription"] = []
    rows["ids"] = []
    rows["target"] = []
    rows["lookupCats"] = {}
    rows["lookupSubCats"] = {}

    t0 = time.time()
    cleanUpData(blocked, sampleSize, rows)
    cleanUpData(unblocked, sampleSize, rows)
    t1 = time.time()
    print "Done sampling, took ", (t1-t0), " Sample size: ", sampleSize

    del blocked
    del unblocked

#    t0 = time.time()
#    wordVectorizer = TfidfVectorizer(ngram_range=(1,1), analyzer="word", binary=False, min_df=2, smooth_idf=True, sublinear_tf=True, use_idf=True, max_features=50000, max_df=0.9, lowercase=True)
#    trainTfIdf = wordVectorizer.fit_transform(rows["textDescription"])
#    t1 = time.time()
#    print "vectorized. Took ", (t1-t0)

    #### Testing
    testing = False
    if testing:
        testVec = []
#        testIds = []
#        testTargets = []
#        numericFeatures = []
#
#        target = numpy.array(target)
#        numericFeats = numpy.array(numericFeats, dtype=float)
#
#        clf = lm.LogisticRegression(penalty='l1', dual=False, tol=1e-06, C=3, fit_intercept=True,intercept_scaling=0.1)
#        clf = clf.fit(trainTfIdf, target)
#
#        nonencodable = numericFeats[:,[5,6,7,8]]
#        enc = OneHotEncoder()
#        hotEncoded = numpy.array(enc.fit_transform(numericFeats[:,[0,1,2,3,4]]).toarray().tolist())
#        newFeats = numpy.concatenate((hotEncoded, nonencodable), axis=1)
#
#        clfNum = GradientBoostingClassifier(n_estimators=500, random_state=0)
#        clfNum = clfNum.fit(newFeats, target)
#
#        t0 = time.time()
#        testData = loadPickledFile('output/testUnblocked.pkl')
#        t1 = time.time()
#        print "unpicked test file, took ",(t1-t0)
#
#        sampleTextFields(testData, len(testData), testVec, testTargets, numericFeatures, testIds, categoryLookup, subCatLookup, False)
#        testTfIdfs = wordVectorizer.transform(testVec)
#
#        numericFeatures = numpy.array(numericFeatures)
#        nonencodableTest = numericFeatures[:,[5,6,7,8]]
#        hotEncodedTest = numpy.array(enc.fit_transform(numericFeatures[:,[0,1,2,3,4]]).toarray().tolist())
#        newFeatsTest = numpy.concatenate((hotEncodedTest, nonencodableTest), axis=1)
#
#        numerif_probas = clfNum.predict_proba(newFeatsTest)[:,1]
#
#        predics_probas = clf.predict_proba(testTfIdfs)[:,1]
#        zippedWeights = zip(predics_probas, numerif_probas)
#        sumed = [(entry[0]*0.8 + entry[1]*0.2) for entry in zippedWeights]
#
#        zippedProbaWeight = sorted(zip(sumed, testIds), reverse=True)
#
#        f = open('output/predict3_0.8.txt','w')
#        for entry in zippedProbaWeight:
#            line = str(entry[1]) + "\n"
#            f.write(line)
#        f.close()
    else:
        t0 = time.time()

        target = numpy.array(rows["target"])
        numericFeats = numpy.array(rows["numericDescription"], dtype=float)
        rows["textDescription"] = numpy.array(rows["textDescription"])

        rows["categoryDescription"] = encodeCategoricalStringFeats(rows["categoryDescription"])
        rows["subcategoryDescription"] = encodeCategoricalStringFeats(rows["subcategoryDescription"])

        scores = []
        # split into train and test
        rs = cross_validation.ShuffleSplit(len(target), 5, test_size=0.2)
        for trainIndex, testIndex in rs:
            wordVectorizer = TfidfVectorizer(ngram_range=(1,1), analyzer="word", binary=False, min_df=2, smooth_idf=True, sublinear_tf=True, use_idf=True, max_features=50000, max_df=0.9, lowercase=True)
            trainTfIdf = wordVectorizer.fit_transform(rows["textDescription"][trainIndex])
            testTfIdf = wordVectorizer.transform(rows["textDescription"][testIndex])
            t1 = time.time()
            print "vectorized. Took ", (t1-t0)

            print testTfIdf[-1]
            # text features
            clf = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0000001, C=3, fit_intercept=True,intercept_scaling=0.1)
            clf = clf.fit(trainTfIdf, target[trainIndex])
            wordsTestPredict = clf.predict_proba(testTfIdf)[:,1]
            stacked = zip(wordsTestPredict, wordsTestPredict)

#            # **** numeric features
             # one hot encode all but the price
            nonencodable = numericFeats[:,[5,6,7,8]]
            enc = OneHotEncoder()
            hotEncoded = numpy.array(enc.fit_transform(numericFeats[:,[0,1,2,3,4]]).toarray().tolist())
            newFeats = numpy.concatenate((hotEncoded, nonencodable), axis=1)
#
            clfNum = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0000001, C=3, fit_intercept=True,intercept_scaling=0.1) #GradientBoostingClassifier(n_estimators=100, random_state=0)
            numericFeatsPredict = clfNum.fit(newFeats[trainIndex,:], target[trainIndex]).predict_proba(newFeats[testIndex,:])[:,1]

            stacked = zip(numericFeatsPredict, numericFeatsPredict)
            #stacked = zip(wordPredict, numericFeatsPredict)
            #stacked = zip(wordPredict, wordPredict)

            #sumed = [1 if ((entry[0]*0.8 + entry[1]*0.2) >= 0.5) else 0 for entry in stacked]
            sumed = [numpy.round(entry[0]) for entry in stacked]
            score = precision_score(target[testIndex], sumed)
            scores.append(score)
            print "Learned ...", score


        print scores
