import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import sklearn.linear_model as lm
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import linear_model
from sklearn import svm
from sklearn.linear_model import RidgeClassifier

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

            if row['is_proved'] not in countofApproved:
                countofApproved[row['is_proved']] = 0
            countofApproved[row['is_proved']] += 1

            if row['close_hours'].strip() != '':
                durationOfAdd.append(float(row['close_hours'])*60)


        features["numericDescription"].append(arr)



    #
    # print countofApproved
    # durationOfAdd = numpy.array(durationOfAdd)
    # print "duration mean: ", numpy.mean(durationOfAdd)
    # print "duration stddev: ", numpy.std(durationOfAdd)
    # print "duration median: ", numpy.median(durationOfAdd)


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

def evaluate(predict, truth, name, is_proba=True):
    rounded = predict
    if is_proba:
        rounded = [numpy.round(entry) for entry in predict]
    score = precision_score(truth, rounded)
    print "Using ",name, " score: ", score
    return score

def loadData():
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

    return rows

def train(rows):
    print "Training ... "
    t0 = time.time()

    target = numpy.array(rows["target"])
    numericFeats = numpy.array(rows["numericDescription"], dtype=float)
    rows["textDescription"] = numpy.array(rows["textDescription"])

    scores = []
    # split into train and test
    rs = cross_validation.ShuffleSplit(len(target), 5, test_size=0.2)
    for trainSetIndex, holdoutIndex in rs:

        # 0. get the data splits, here test is split into train and test to generate input for the secondary model
        trainIndex, testIndex = train_test_split(trainSetIndex, test_size=0.2, random_state=42)

        # 1. text based features
        wordVectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer="word", binary=False, min_df=2, smooth_idf=True, sublinear_tf=True, use_idf=True, max_features=50000, max_df=0.9, lowercase=True)
        trainTfIdf = wordVectorizer.fit_transform(rows["textDescription"][trainIndex])
        testTfIdf = wordVectorizer.transform(rows["textDescription"][testIndex])

        # model text features
        clf1 = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0000001, C=3, fit_intercept=True,intercept_scaling=0.1).fit(trainTfIdf, target[trainIndex])
        wordsTestPredictLR = clf1.predict_proba(testTfIdf)[:,1]
        evaluate(wordsTestPredictLR, target[testIndex], name="wordsTestPredictLR")

        clf2 = linear_model.SGDClassifier(penalty='l2', n_iter=1200, alpha=0.000001,n_jobs=2).fit(trainTfIdf, target[trainIndex])
        wordsTestPredictSGD = clf2.predict(testTfIdf)
        evaluate(wordsTestPredictSGD, target[testIndex], name="wordsTestPredictSGD")

        clf3 = svm.LinearSVC().fit(trainTfIdf, target[trainIndex])
        wordsTestPredictSVC = clf3.predict(testTfIdf)
        evaluate(wordsTestPredictSVC, target[testIndex], name="wordsTestPredictSVC", is_proba=False)

        clf4 = RidgeClassifier(tol=0.000001, normalize=True).fit(trainTfIdf, target[trainIndex])
        wordsTestPredictRidge = clf4.predict(testTfIdf)
        evaluate(wordsTestPredictRidge, target[testIndex], name="wordsTestPredictRidge", is_proba=False)


        # # 2. numeric features, some need to be one hot encoded some don't
        # nonencodable = numericFeats[:,[5,6,7,8]]
        # enc = OneHotEncoder()
        # hotEncoded = numpy.array(enc.fit_transform(numericFeats[:,[0,1,2,3,4]]).toarray().tolist())
        # newFeats = numpy.concatenate((hotEncoded, nonencodable), axis=1)
        #
        # # model numeric features
        # clfNum1 = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0000001, C=3, fit_intercept=True,intercept_scaling=0.1).fit(newFeats[trainIndex,:], target[trainIndex])
        # numericFeatsPredictLR = clfNum1.predict_proba(newFeats[testIndex,:])[:,1]
        # evaluate(numericFeatsPredictLR, target[testIndex], name="numericFeatsPredictLR")
        #
        # clfNum2 = RandomForestClassifier(200).fit(newFeats[trainIndex,:], target[trainIndex])
        # numericFeatsPredictRF = clfNum2.predict_proba(newFeats[testIndex,:])[:,1]
        # evaluate(numericFeatsPredictRF, target[testIndex], name="numericFeatsPredictRF")
        #
        # #clfNum3 = GradientBoostingClassifier(100).fit(newFeats[trainIndex,:], target[trainIndex])
        # #numericFeatsPredictGBT = clfNum3.predict_proba(newFeats[testIndex,:])[:,1]
        #
        # clfNum4 = svm.LinearSVC().fit(newFeats[trainIndex,:], target[trainIndex])
        # numericFeatsPredictSVC = clfNum4.predict(newFeats[testIndex,:])
        # evaluate(numericFeatsPredictSVC, target[testIndex], name="numericFeatsPredictSVC", is_proba=False)



        #3. numeric feats + textfeats => to generate level2 model
        featsModelLevel2 = numpy.column_stack((wordsTestPredictLR, wordsTestPredictSGD, wordsTestPredictSVC, wordsTestPredictRidge))#, numericFeatsPredictRF, numericFeatsPredictSVC))

        # train on previous steps predict's target
        #clfLevel2 = GradientBoostingClassifier(n_estimators=100, random_state=0)
        clfLevel2 = lm.LogisticRegression().fit(featsModelLevel2, target[testIndex])
        clfLevel21 = GradientBoostingClassifier(n_estimators=500, random_state=0).fit(featsModelLevel2, target[testIndex])

        lvlTwoTstTfIdf = wordVectorizer.transform(rows["textDescription"][holdoutIndex])


        #4. now take the test inputs from the holdout set and train using the initial
        #text and numeric models
        wordsTestPredictLRHoldOut = clf1.predict_proba(lvlTwoTstTfIdf)[:,1]
        wordsTestPredictSGDHoldOut = clf2.predict(lvlTwoTstTfIdf)
        wordsTestPredictSVCHoldOut = clf3.predict(lvlTwoTstTfIdf)
        wordsTestPredictRidgeHoldOut = clf4.predict(lvlTwoTstTfIdf)


        # # and now for numeric features
        # numericFeatsPredictLRHoldOut = clfNum1.predict_proba(newFeats[holdoutIndex,:])[:,1]
        # numericFeatsPredictRFHoldOut = clfNum2.predict_proba(newFeats[holdoutIndex,:])[:,1]
        # #numericFeatsPredictGBTHoldOut = clfNum3.predict_proba(newFeats[holdoutIndex,:])[:,1]
        # numericFeatsPredictSVCHoldOut = clfNum4.predict(newFeats[holdoutIndex,:])


        # create a feature set whose target is target{testIndex}
        testFeatsLevelTwo = numpy.column_stack((wordsTestPredictLRHoldOut, wordsTestPredictSGDHoldOut, wordsTestPredictSVCHoldOut, wordsTestPredictRidgeHoldOut)) #, numericFeatsPredictRFHoldOut, numericFeatsPredictSVCHoldOut))

        finalPredict = clfLevel2.predict_proba(testFeatsLevelTwo)[:,1]
        evaluate(finalPredict, target[holdoutIndex], name="Level2LR", is_proba=True)

        finalPredictGBM = clfLevel21.predict_proba(testFeatsLevelTwo)[:,1]
        evaluate(finalPredictGBM, target[holdoutIndex], name="Level2GBM", is_proba=True)



    print scores


if __name__=="__main__":

    rows = loadData()



    #### Testing
    testing = False
    if testing:
        testVec = []

        numericFeats = numpy.array(rows["numericDescription"], dtype=float)
        rows["textDescription"] = numpy.array(rows["textDescription"])
        target = numpy.array(rows["target"])

        scores = []
        # split into train and test
        trIndex, tsIndex = []
        rs = cross_validation.ShuffleSplit(len(target), n_iter=1,test_size=.25, random_state=0)

        for tr,ts in rs:
            trIndex = tr
            tsIndex = ts

        wordVectorizer = TfidfVectorizer(ngram_range=(1,1), analyzer="word", binary=False, min_df=2, smooth_idf=True, sublinear_tf=True, use_idf=True, max_features=50000, max_df=0.9, lowercase=True)

        # train first model by using the training set
        trainTfIdf = wordVectorizer.fit_transform(rows["textDescription"][trIndex])
        testTfIdf = wordVectorizer.transform(rows["textDescription"][tsIndex])
        t1 = time.time()


        # text features
        clf = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0000001, C=3, fit_intercept=True,intercept_scaling=0.1)
        clf = clf.fit(trainTfIdf, target[trIndex])
        wordsTestPredict = clf.predict_proba(testTfIdf)[:,1]


        # **** numeric features
        # one hot encode all but the price
        nonencodable = numericFeats[:,[5,6,7,8]]
        enc = OneHotEncoder()
        hotEncoded = numpy.array(enc.fit_transform(numericFeats[:,[0,1,2,3,4]]).toarray().tolist())
        newFeats = numpy.concatenate((hotEncoded, nonencodable), axis=1)

        clfNum = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0000001, C=3, fit_intercept=True,intercept_scaling=0.1) #GradientBoostingClassifier(n_estimators=100, random_state=0)
        clfNum = clfNum.fit(newFeats[trIndex,:], target[trIndex])
        numericFeatsPredict = clfNum.predict_proba(newFeats[tsIndex,:])[:,1]

        ## numeric feats + textfeats => target[tsIndex] train level 2 model
        featsModelLevel2 = numpy.column_stack((wordsTestPredict,numericFeatsPredict))
        clfLevel2 = GradientBoostingClassifier(n_estimators=500, random_state=0)
        #clfLevel2 = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0000001, C=3, fit_intercept=True,intercept_scaling=0.1)
        clfLevel2 = clfLevel2.fit(featsModelLevel2, target[tsIndex])





        testData = loadPickledFile('output/testUnblocked.pkl')
        testrows = {}

        testrows["textDescription"] = [];
        testrows["numericDescription"] = [];
        testrows["categoryDescription"] = [];
        testrows["subcategoryDescription"] = []
        testrows["ids"] = []
        testrows["target"] = []
        testrows["lookupCats"] = rows["lookupCats"]
        testrows["lookupSubCats"] = rows["lookupSubCats"]

        cleanUpData(testData, len(testData), testrows, False)

        testrows["numericDescription"] = numpy.array(testrows["numericDescription"], dtype=float)
        testrows["textDescription"] = numpy.array(testrows["textDescription"])

        tstTfIdf = wordVectorizer.transform(testrows["textDescription"])
        tstWordPredict = clf.predict_proba(tstTfIdf)[:,1]

        hotencodedtst = enc.transform(testrows["numericDescription"][:,[0,1,2,3,4]]).toarray().tolist();
        newTstFeats = numpy.concatenate((hotencodedtst, testrows["numericDescription"][:,[5,6,7,8]]), axis=1)
        numericFeatsTstPredict = clfNum.predict_proba(newTstFeats)[:,1]

        ## numeric feats + textfeats => target[tsIndex] train level 2 model
        lvl2Feats = numpy.column_stack((tstWordPredict,numericFeatsTstPredict))
        lvl2Predict = clfLevel2.predict_proba(lvl2Feats)[:,1]

        zippedProbaWeight = sorted(zip(lvl2Predict, testrows["ids"]), reverse=True)

        f = open('output/predict3_0.8.txt','w')
        for entry in zippedProbaWeight:
            line = str(entry[1]) + "\n"
            f.write(line)
        f.close()



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
        train(rows)
        print "hello"