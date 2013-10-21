# -*- coding: utf-8 -*-
"""
Created on Tue Sep 03 11:10:13 2013

@author: ytsegay
"""

import csv
import numpy as np
import json
from sklearn.naive_bayes import MultinomialNB
import sklearn.linear_model as lm
from time import time


#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import cross_val_score
from sklearn.metrics import f1_score


from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier



from nltk import word_tokenize          
from nltk.stem import PorterStemmer
import nltk
import string

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = PorterStemmer()
    def __call__(self, doc):
        return [self.wnl.stem(t) for t in word_tokenize(doc)]


def everything_between(content, begin="<body", end="</body>"):
    idx1=content.find(begin)
    idx2=content.find(end,idx1)+len(end)
    return content[idx1:idx2].strip()

def processText(inString):
    inString = nltk.clean_html(inString)
    inString = inString.replace("\t", " ")
    inString = inString.replace("\r", " ")
    inString = inString.replace("\n", " ")
    
    for punct in string.punctuation:
        inString = inString.replace(punct," ")
    
    return inString
    
def getRawData(fileName):
    
    content = open(fileName).read().lower()
    content = content.decode("ISO-8859-1")
    body=everything_between(content)
    docTitle = everything_between(content, begin="<title", end="</title>")
    
    if body == '':
        body = content
        
    body = processText(body)
    docTitle = processText(docTitle)
    
     # get metadata 
    docMeta = ''
    return docTitle,body,docMeta


def loadData(fileName="data\\train.tsv", withLabel=True):
    csvFile = open(fileName, 'rb')
    csvRows = csv.reader(csvFile, delimiter='\t')    
    
    linesCounter = 0
    categories = {}
    categoriesCounter = 0
    docsText = []
    docsTitle = []
    docsUrl = []
    rows = []
    docsMeta = []
    
    countOfSmallBody = 0
    howmanywords = 5000
    for row in csvRows:
        # skip header 
        # keep the title
        if linesCounter > 0:
            docTitle = u''
            docText = u''
            docUrl = u''
            docMeta = u''
        
            jsonItems = json.loads(row[2]).items()
            if len(jsonItems) > 1:
                for i in range(len(jsonItems)):
                    if (jsonItems[i][0] == 'title') and len(jsonItems[i]) > 0 and jsonItems[i][1] != None: 
                        docTitle = jsonItems[i][1]
                    elif (jsonItems[i][0] == 'body') and len(jsonItems[i]) > 0 and jsonItems[i][1] != None:
                        docInWords = jsonItems[i][1].split()
                        docText = ' '.join(docInWords[0:howmanywords])
                    elif (jsonItems[i][0] == 'url') and len(jsonItems[i]) > 0 and jsonItems[i][1] != None:
                        docUrl = jsonItems[i][1]
            
            # if the body failed go and get the text and title from the raw doc  
            tmpDocMeta = '' #getMetaData("".join(['data\\raw_content\\raw_content\\',row[1]]))
            if (len(docText.split()) < 10):
                tmpDocTitle, tmpDocText,tmpDocMeta = getRawData("".join(['data\\raw_content\\raw_content\\',row[1]]))
                docTitle,docText = tmpDocTitle, tmpDocText
                docInWords = docText.split()                
                docText = ' '.join(docInWords[0:howmanywords])
            
            
            # lets keep docs with no valid text/valid raw doc from the training
            #if (len(docText.split()) < 10):
            #    countOfSmallBody = countOfSmallBody + 1                                
            #    #print row[1],"\t",row[-1];
            #    #print docText
            #
            #    #appendRow = False
            
            appendRow = True
            if appendRow == True:
                docsText.append(docText)
                docsTitle.append(docTitle)
                docsUrl.append(docUrl)
                docsMeta.append(tmpDocMeta)
    
            
            # removing text
            del row[2]
            # remove url as it is already represented by the id
            del row[0]

            if row[1] == '?':
                row[1] = 'unknown'
                
            # cols 2, 15 and 18 seem to have missing data in both test and
            # and training
            if row[2] == '?':
                row[2] = 0
            
            # is news
            if row[15] == '?':
                row[15] = 0
                
            # news front page?
            if row[18] == '?':
                row[18] = 0.5
            
            # image ratio cannot be less than 0
            if row[14] == '-1':
                row[14] = 0
            
            #embed ratio cannot be less than 0
            if row[9] == '-1':
                row[9] = 0
            
            if row[3] > 10:
                row[3] = 10

            #busCount = 0
            if not categories.has_key(row[1]):
                categoriesCounter = categoriesCounter + 1
                categories[row[1]] = categoriesCounter
                
            # replace category with id for one hot encoding
            row[1] = categories[row[1]]
            
            rows.append(row)

        linesCounter = linesCounter + 1;
    print "Docs with little text : ",countOfSmallBody," of ", len(rows)
    return rows,docsTitle,docsUrl,docsText,docMeta

def combineTxt(feat1, feat2, feat3=None):
    comb = []
    if np.shape(feat1) == np.shape(feat2):
        for i in range(len(feat1)):
            newLine = feat1[i] + " " + feat2[i]
            if feat3 != None:
                newLine = newLine + " " + feat3[i]
            
            comb.append(newLine)

    return comb

def textClassify():
    featsTrain, titlesTrain,urlsTrain,bodiesTrain,metasTrain = loadData("data\\train.tsv")
    
    txtFeats = {"npFeatsTrain": np.array(featsTrain, dtype=np.float32),
                "npTitlesTrain": titlesTrain,
                "npUrlsTrain": urlsTrain,
                "npBodiesTrain": bodiesTrain
                }

    txtFeats["npTitleAndUrl"] = combineTxt(txtFeats["npUrlsTrain"], txtFeats["npTitlesTrain"])
    txtFeats["npTitleAndBody"] = combineTxt(txtFeats["npTitlesTrain"], txtFeats["npBodiesTrain"])
    txtFeats["npUrlAndBody"] = combineTxt(txtFeats["npUrlsTrain"], txtFeats["npBodiesTrain"])
    txtFeats["*npTitleBodyAndUrl*"] = combineTxt(txtFeats["npTitlesTrain"], txtFeats["npUrlsTrain"], txtFeats["npBodiesTrain"], )

    minDf = 2
    maxFeat = 5000    
    vectorizers = {"npFeatsTrain": None,
                "npTitlesTrain": TfidfVectorizer(min_df=minDf, ngram_range=(1, 1), stop_words='english', max_features=maxFeat),
                "npUrlsTrain": TfidfVectorizer(min_df=minDf, max_features=maxFeat, sublinear_tf=1),
                "npTitleAndUrl": TfidfVectorizer(min_df=minDf, max_features=maxFeat, sublinear_tf=1),
                "npBodiesTrain": TfidfVectorizer(min_df=minDf, max_features=maxFeat, sublinear_tf=1),
                "npTitleAndBody": TfidfVectorizer(min_df=minDf, max_features=maxFeat, sublinear_tf=1),
                "npUrlAndBody"  : TfidfVectorizer(min_df=minDf, max_features=maxFeat, sublinear_tf=1),
                "*npTitleBodyAndUrl*": TfidfVectorizer(min_df=minDf, max_features=maxFeat, smooth_idf=1,sublinear_tf=True)
            }

    # seperate target from the trainSet
    npTrainTarget = txtFeats["npFeatsTrain"][:,24]
    
    for key, value in txtFeats.iteritems():
        if (key != 'npFeatsTrain'):
            print key

            vectorizer = vectorizers[key]
            vects = vectorizer.fit_transform(value).toarray()
            
            #clf = MultinomialNB(alpha=0.00001)
            clf = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                            C=1, fit_intercept=True, intercept_scaling=1.0)

            scores = cross_val_score(clf, vects, y=npTrainTarget, cv=10, score_func=f1_score)

            
            print "Scores: ", scores
            print "Average: ", np.mean(scores)
            print ""
            print ""
            



























def oneHotEncode(npTrainRows, npTestRows, indices=None):
    
    merged = np.vstack((npTrainRows[:,0:24], npTestRows))
    
    for index in indices:
        enc = OneHotEncoder()
        enc.fit(np.transpose([merged[:,index]]))
        
        encodedTrainCol = enc.transform(np.transpose([npTrainRows[:,index]])).toarray()
        npTrainRows = np.hstack((npTrainRows, encodedTrainCol))
        
    return npTrainRows


def featuresClassify():
    featsTrain, titlesTrain,urlsTrain,bodiesTrain = loadData("data\\train.tsv")
    npFeatsTrain = np.array(featsTrain, dtype=np.float32)

    featsTest, titlesTest,urlsTest,bodiesTest = loadData("data\\test.tsv")
    npFeatsTest = np.array(featsTest, dtype=np.float32)

    #print npFeatsTrain[:,17]
    #,12,16,18
    npFeatsTrain = oneHotEncode(npFeatsTrain, npFeatsTest, [1,9,15,16,18])

    
  
    # seperate target from the trainSet
    npTrainTarget = npFeatsTrain[:,24]
    
    npFeatsTrain = np.delete(npFeatsTrain, [0,10,24,1,9,15,16,18], axis=1)

    t1 = time()
    #clf = SVC(kernel='linear', C=1)
    params = {#'n_estimators':[100,200,300,400,500], 
              'criterion':['gini', 'entropy'],
              'max_depth': [4,5,6,7,8,9], 
              'max_features':[None, 'log2', 'auto']
              }
    
    #clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params, score_func=f1_score, n_jobs=2)

    #clf = RandomForestClassifier()
    #scores = cross_val_score(clf, npFeatsTrain, y=npTrainTarget, cv=10, score_func=f1_score)
    #clf = clf.fit(npFeatsTrain, y=npTrainTarget)
    #print "estimators best: ", clf.best_estimator_

    
#    isC = False
#    if isC:    
#        clf = RandomForestClassifier(n_estimators=250, compute_importances=True)
#        clf.fit(npFeatsTrain, npTrainTarget)
#        
#        importances = clf.feature_importances_
#        print clf.feature_importances_
#        std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
#        indices = np.argsort(importances)[::-1]
#        
#        print("Feature ranking:")
#    
#        for f in range(len(indices)):
#            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#        print std
#        print npFeatsTrain[0]


    clfs = {"RandomForest":RandomForestClassifier(n_estimators=250), 
            "GradientBoost":GradientBoostingClassifier(n_estimators=250), 
            "ExtraTrees":ExtraTreesClassifier(n_estimators=250)}
            
    for name,clf in clfs.items():
        scores = cross_val_score(clf, npFeatsTrain, y=npTrainTarget, cv=10, score_func=f1_score)    
        print "name: ", name
        print "Scores: ", scores
        print "Average: ", np.mean(scores)
        print "\n\n"

    t2 = time()
    print "\n\nTime taken: ", (t2-t1)

    print ""
    print ""
    
if __name__ == '__main__':
    textClassify()
    #featuresClassify()