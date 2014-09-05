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
import sklearn.svm as svm

import numpy
import time
from sklearn.preprocessing import OneHotEncoder


# used to map targets. Adds a separate class in the case a document is_proven (and that is passed)
def target_mapper(target, is_proved_illicit=u'0'):
    if target == u'1':
        if is_proved_illicit == u'1':
            return 2
        return 1
    return 0


#preps data to be used by vectorizer and classifier
def prep_data(data_rows, sample_size, features, has_target=True):

    sample = random.sample(xrange(len(data_rows)), sample_size)
    for index in sample:
        data_row = data_rows[index]

        #ad id
        features["ids"].append(data_row["itemid"])

        #textual work
        # concatenate attribute keys and values into a single string
        attributes = ''
        for key in data_row['attrs']:
            attributes += " " + key + " " + data_row['attrs'][key]
        attributes = attributes.strip()

        # combine title + description and attrs into one text field
        text = data_row['title'] + " " + data_row['description'] + " " + attributes + " "
        features["textDescription"].append(text)

        # numeric features
        arr = []

        # does the list contain a phone #?
        has_phone = 0
        if int(data_row['phones_cnt']) > 0:
            has_phone = 1
        arr.append(has_phone)

        # boolean if it has an email
        has_email = 0
        if int(data_row['emails_cnt']) > 0:
            has_email = 1
        arr.append(has_email)

        # boolean attribute if it has a URL
        has_urls = 0
        if int(data_row['urls_cnt']) > 0:
            has_urls = 1
        arr.append(has_urls)

        # encode category and sub category
        # todo: find a better hashing method, rather than a dictionary lookup
        entry = data_row['subcategory']
        if entry not in features["lookupSubCats"]:
            features["lookupSubCats"][entry] = len(features["lookupSubCats"])
        arr.append(features["lookupSubCats"][entry])

        entry = data_row['category']
        if entry not in features["lookupCats"]:
            features["lookupCats"][entry] = len(features["lookupCats"])
        arr.append(features["lookupCats"][entry])

        arr.append(float(data_row['price']))

        # additional feature count words in title
        title_length = len(data_row['title'].split())
        arr.append(title_length)

        # count of words in the description
        text_length = len(text.split())
        arr.append(text_length)

        # depth of the attributes
        arr.append(len(data_row['attrs']))



        if has_target:
            features["target"].append(target_mapper(data_row['is_blocked'], data_row['is_proved']))

            # if row['is_proved'] not in countofApproved:
            #     countofApproved[row['is_proved']] = 0
            # countofApproved[row['is_proved']] += 1
            #
            # if row['close_hours'].strip() != '':
            #     durationOfAdd.append(float(row['close_hours'])*60)

        features["numericDescription"].append(arr)


def load_pickled_file(fileName):
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


def load_data(train_mode=True):
    data_rows = {}
    if train_mode:
        t0 = time.time()
        blocked = load_pickled_file('output/trainblocked.pkl')
        t1 = time.time()
        print "unpickled blocked, took ",(t1-t0)

        t0 = time.time()
        unblocked = load_pickled_file('output/trainunblocked.pkl')
        t1 = time.time()
        print "unpickled unblocked, took ",(t1-t0)

        # blocked are always smaller so will use them to make sure we get same size samples
        # for both classes
        sample_size = len(blocked)

        data_rows["textDescription"] = [];
        data_rows["numericDescription"] = [];
        data_rows["categoryDescription"] = [];
        data_rows["subcategoryDescription"] = []
        data_rows["ids"] = []
        data_rows["target"] = []
        data_rows["lookupCats"] = {}
        data_rows["lookupSubCats"] = {}

        t0 = time.time()
        prep_data(blocked, sample_size, data_rows)
        prep_data(unblocked, sample_size, data_rows)
        t1 = time.time()
        print "Done sampling, took ", (t1-t0), " Sample size: ", sample_size

        # free up some memory
        del blocked
        del unblocked
    else:
        t0 = time.time()
        tst_data = load_pickled_file('output/testUnblocked.pkl')
        t1 = time.time()
        print "unpickled test, took ",(t1-t0)

        data_rows["textDescription"] = []
        data_rows["numericDescription"] = []
        data_rows["categoryDescription"] = []
        data_rows["subcategoryDescription"] = []
        data_rows["ids"] = []
        data_rows["target"] = []
        data_rows["lookupCats"] = data_rows["lookupCats"]
        data_rows["lookupSubCats"] = data_rows["lookupSubCats"]

        prep_data(tst_data, len(tst_data), data_rows, False)

        data_rows["textDescription"] = numpy.array(data_rows["textDescription"])
        # done so free it up
        del tst_data

    return data_rows

def test(train_rows, test_rows):
    # need to load training data

    print "Testing ..."
    train_rows["textDescription"] = numpy.array(train_rows["textDescription"])
    target = numpy.array(train_rows["target"])

    word_vectorizer = TfidfVectorizer(ngram_range=(1,4), analyzer="word", binary=False, min_df=2, smooth_idf=True,
                                      sublinear_tf=True, use_idf=True, max_features=50000, max_df=0.9, lowercase=True)
    train_tf_idf = word_vectorizer.fit_transform(train_rows["textDescription"])

    clf = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0000001, C=3, fit_intercept=True,
                                intercept_scaling=0.1).fit(train_tf_idf, target)

    tst_tf_idf = word_vectorizer.transform(test_rows["textDescription"])

    tst_word_predict = clf.predict_proba(tst_tf_idf)[:,1]

    zipped_proba_weight = sorted(zip(tst_word_predict, test_rows["ids"]), reverse=True)

    f = open("output/predict_1_4grams_LR.txt",'w')
    for entry in zipped_proba_weight:
        line = str(entry[1]) + "\n"
        f.write(line)
    f.close()


def fit_and_evaluate(clf, train_x, train_y, test_x, test_y, is_proba=True):
    clf = clf.fit(train_x, train_y)
    results = []
    if is_proba:
        results = clf.predict_proba(test_x)[:, 1]
    else:
        results = clf.predict(test_x)
    evaluate(results, test_y, name="name", is_proba=False)
    return results


def train(train_rows):
    print "Training ... "
    t0 = time.time()

    target = numpy.array(train_rows["target"])
    numeric_feats = numpy.array(train_rows["numericDescription"], dtype=float)
    train_rows["textDescription"] = numpy.array(train_rows["textDescription"])

    scores = []
    # split into train and test
    rs = cross_validation.ShuffleSplit(len(target), 3, test_size=0.2)
    for train_set_indices, holdout_indices in rs:

        # 0. get the data splits, here test is split into train and test to generate input for the secondary model
        train_indices, test_indices = train_test_split(train_set_indices, test_size=0.2)


        # 1. text based features
        word_vectorizer = TfidfVectorizer(ngram_range=(1,4), analyzer="word", binary=False, min_df=2,
                                          smooth_idf=True, sublinear_tf=True, use_idf=True, max_features=50000,
                                          max_df=0.9, lowercase=True)
        train_tf_idf = word_vectorizer.fit_transform(train_rows["textDescription"][train_indices])
        test_tf_idf = word_vectorizer.transform(train_rows["textDescription"][test_indices])

        # model text features
        clf1 = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0000001, C=3, fit_intercept=True,
                                     intercept_scaling=0.1)

        clf2 = lm.SGDClassifier(penalty='l2', n_iter=1200, alpha=0.000001, n_jobs=2)

        clf3 = svm.LinearSVC(C=1, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, loss='l2',
                             multi_class='ovr', penalty='l1',random_state=None, tol=0.001, verbose=0)

        clf1_predict = fit_and_evaluate(clf1, train_tf_idf, target[train_indices], test_tf_idf, target[test_indices])
        clf2_predict = fit_and_evaluate(clf2, train_tf_idf, target[train_indices], test_tf_idf, target[test_indices])
        clf3_predict = fit_and_evaluate(clf3, train_tf_idf, target[train_indices], test_tf_idf, target[test_indices], False)


        # model 3

    #     clf4 = RidgeClassifier(tol=0.000001, normalize=True).fit(trainTfIdf, target[trainIndex])
    #     wordsTestPredictRidge = clf4.predict(testTfIdf)
    #     evaluate(wordsTestPredictRidge, target[testIndex], name="wordsTestPredictRidge", is_proba=False)
    #
    #
    #     # # 2. numeric features, some need to be one hot encoded some don't
    #     # nonencodable = numericFeats[:,[5,6,7,8]]
    #     # enc = OneHotEncoder()
    #     # hotEncoded = numpy.array(enc.fit_transform(numericFeats[:,[0,1,2,3,4]]).toarray().tolist())
    #     # newFeats = numpy.concatenate((hotEncoded, nonencodable), axis=1)
    #     #
    #     # # model numeric features
    #     # clfNum1 = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0000001, C=3, fit_intercept=True,intercept_scaling=0.1).fit(newFeats[trainIndex,:], target[trainIndex])
    #     # numericFeatsPredictLR = clfNum1.predict_proba(newFeats[testIndex,:])[:,1]
    #     # evaluate(numericFeatsPredictLR, target[testIndex], name="numericFeatsPredictLR")
    #     #
    #     # clfNum2 = RandomForestClassifier(200).fit(newFeats[trainIndex,:], target[trainIndex])
    #     # numericFeatsPredictRF = clfNum2.predict_proba(newFeats[testIndex,:])[:,1]
    #     # evaluate(numericFeatsPredictRF, target[testIndex], name="numericFeatsPredictRF")
    #     #
    #     # #clfNum3 = GradientBoostingClassifier(100).fit(newFeats[trainIndex,:], target[trainIndex])
    #     # #numericFeatsPredictGBT = clfNum3.predict_proba(newFeats[testIndex,:])[:,1]
    #     #
    #     # clfNum4 = svm.LinearSVC().fit(newFeats[trainIndex,:], target[trainIndex])
    #     # numericFeatsPredictSVC = clfNum4.predict(newFeats[testIndex,:])
    #     # evaluate(numericFeatsPredictSVC, target[testIndex], name="numericFeatsPredictSVC", is_proba=False)
    #
    #
    #
    #     #3. numeric feats + textfeats => to generate level2 model
    #     featsModelLevel2 = numpy.column_stack((wordsTestPredictLR, wordsTestPredictSGD, wordsTestPredictSVC, wordsTestPredictRidge))#, numericFeatsPredictRF, numericFeatsPredictSVC))
    #
    #     # train on previous steps predict's target
    #     #clfLevel2 = GradientBoostingClassifier(n_estimators=100, random_state=0)
    #     clfLevel2 = lm.LogisticRegression().fit(featsModelLevel2, target[testIndex])
    #     #clfLevel21 = GradientBoostingClassifier(n_estimators=500, random_state=0).fit(featsModelLevel2, target[testIndex])
    #
    #     lvlTwoTstTfIdf = wordVectorizer.transform(rows["textDescription"][holdoutIndex])
    #
    #
    #     #4. now take the test inputs from the holdout set and train using the initial
    #     #text and numeric models
    #     wordsTestPredictLRHoldOut = clf1.predict_proba(lvlTwoTstTfIdf)[:,1]
    #     wordsTestPredictSGDHoldOut = clf2.predict(lvlTwoTstTfIdf)
    #     wordsTestPredictSVCHoldOut = clf3.predict(lvlTwoTstTfIdf)
    #     wordsTestPredictRidgeHoldOut = clf4.predict(lvlTwoTstTfIdf)
    #
    #
    #     # # and now for numeric features
    #     # numericFeatsPredictLRHoldOut = clfNum1.predict_proba(newFeats[holdoutIndex,:])[:,1]
    #     # numericFeatsPredictRFHoldOut = clfNum2.predict_proba(newFeats[holdoutIndex,:])[:,1]
    #     # #numericFeatsPredictGBTHoldOut = clfNum3.predict_proba(newFeats[holdoutIndex,:])[:,1]
    #     # numericFeatsPredictSVCHoldOut = clfNum4.predict(newFeats[holdoutIndex,:])
    #
    #
    #     # create a feature set whose target is target{testIndex}
    #     testFeatsLevelTwo = numpy.column_stack((wordsTestPredictLRHoldOut, wordsTestPredictSGDHoldOut, wordsTestPredictSVCHoldOut, wordsTestPredictRidgeHoldOut)) #, numericFeatsPredictRFHoldOut, numericFeatsPredictSVCHoldOut))
    #
    #     finalPredict = clfLevel2.predict_proba(testFeatsLevelTwo)[:,1]
    #     evaluate(finalPredict, target[holdoutIndex], name="Level2LR", is_proba=True)
    #
    #     #finalPredictGBM = clfLevel21.predict_proba(testFeatsLevelTwo)[:,1]
    #     #evaluate(finalPredictGBM, target[holdoutIndex], name="Level2GBM", is_proba=True)
    #
    #
    #
    # print scores


if __name__=="__main__":

    tr_rows = load_data()

    #### Testing
    testing = False
    if testing:
        tst_rows = load_data(False)
        test(tr_rows, tst_rows)
    else:
        train(tr_rows)
        print "hello"