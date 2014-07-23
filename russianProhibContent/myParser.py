import nltk
from nltk import SnowballStemmer
import csv
from nltk.tokenize import RegexpTokenizer
import random
import pickle

def populateRussianStopWords():
	arr = [word.decode('utf-8') for word in nltk.corpus.stopwords.words("russian") if word != "He"]
	return arr

stopwords = populateRussianStopWords()
stemmer = SnowballStemmer('russian')


def stem_word(str, shouldStem):
	if shouldStem == True:
		return stemmer.stem(str)
	return str

def stop_word(str, shouldStop):
	if (shouldStop):
		if str in stopwords:
			return ""

	return str

def translate_word(w):
	cyrillic_translit={u'\u0410': 'A', u'\u0430': 'a',
		u'\u0411': 'B', u'\u0431': 'b',
		u'\u0412': 'V', u'\u0432': 'v',
		u'\u0413': 'G', u'\u0433': 'g',
		u'\u0414': 'D', u'\u0434': 'd',
		u'\u0415': 'E', u'\u0435': 'e',
		u'\u0416': 'Zh', u'\u0436': 'zh',
		u'\u0417': 'Z', u'\u0437': 'z',
		u'\u0418': 'I', u'\u0438': 'i',
		u'\u0419': 'I', u'\u0439': 'i',
		u'\u041a': 'K', u'\u043a': 'k',
		u'\u041b': 'L', u'\u043b': 'l',
		u'\u041c': 'M', u'\u043c': 'm',
		u'\u041d': 'N', u'\u043d': 'n',
		u'\u041e': 'O', u'\u043e': 'o',
		u'\u041f': 'P', u'\u043f': 'p',
		u'\u0420': 'R', u'\u0440': 'r',
		u'\u0421': 'S', u'\u0441': 's',
		u'\u0422': 'T', u'\u0442': 't',
		u'\u0423': 'U', u'\u0443': 'u',
		u'\u0424': 'F', u'\u0444': 'f',
		u'\u0425': 'Kh', u'\u0445': 'kh',
		u'\u0426': 'Ts', u'\u0446': 'ts',
		u'\u0427': 'Ch', u'\u0447': 'ch',
		u'\u0428': 'Sh', u'\u0448': 'sh',
		u'\u0429': 'Shch', u'\u0449': 'shch',
		u'\u042a': '"', u'\u044a': '"',
		u'\u042b': 'Y', u'\u044b': 'y',
		u'\u042c': "'", u'\u044c': "'",
		u'\u042d': 'E', u'\u044d': 'e',
		u'\u042e': 'Iu', u'\u044e': 'iu',
		u'\u042f': 'Ia', u'\u044f': 'ia'}

	converted_word = ''
	for char in w:
		transchar = ''
		if char in cyrillic_translit:
			transchar = cyrillic_translit[char]
		else:
			transchar = char
		converted_word += transchar
	return converted_word

def normalize_words(words, shouldStem, shouldStop):
	retStr = ""
	for word in words:
		word = stop_word(word, shouldStop)
		word = stem_word(word, shouldStem)

		if (word.strip() != ""):
			retStr += translate_word(word.strip()) + " "
	return retStr.strip()

def load_data(fileName, linesCount, columnsToUse):
	
	cleanedDataBlocked = []
	cleanedDataNotBlocked = []
	countPos = 0
	total = 0
	
	tokenizer = RegexpTokenizer(r'\w+')

	linesCounter = 0
	with open(fileName,'rb') as fl:
		itemReader = csv.DictReader(fl, delimiter='\t', quotechar = '"')
		for i, item in enumerate(itemReader):
			if linesCounter >= 0:
				item = {featureName: featureValue.decode('utf-8') for featureName,featureValue in item.iteritems()}
				for columnToUse in columnsToUse:
					words = tokenizer.tokenize(item[columnToUse])

					item[columnToUse] = normalize_words(words, True, True)
				
				item['attrs'] = ''
				
				if 'is_blocked' in item and item['is_blocked'] == u'1':
					countPos += 1
					cleanedDataBlocked.append(item)
				else:
					cleanedDataNotBlocked.append(item)
				total += 1
			linesCounter += 1
	return cleanedDataNotBlocked,cleanedDataBlocked

	

if __name__=="__main__":
	#outputUnBlocked, outputBlocked = load_data("/Users/ytsegay/dataProjects/kaggle/russianadds/data/train.tsv", 100000, ['category', 'subcategory', 'title', 'description'])
	#sampledOutputBlocked = random.sample(outputUnBlocked, len(outputBlocked))

	#fklFl = open("output/trainunblocked.pkl", "wb")
	#pickle.dump(outputUnBlocked, fklFl);
	#fklFl.close()
	
	#fklFl = open("output/trainblocked.pkl", "wb")
	#pickle.dump(outputBlocked, fklFl);
	#fklFl.close()
	
	outputUnBlocked, outputBlocked = load_data("/Users/ytsegay/dataProjects/kaggle/russianadds/data/test.tsv", 100000, ['category', 'subcategory', 'title', 'description'])
	fklFl = open("output/testUnblocked.pkl", "wb")
	pickle.dump(outputUnBlocked, fklFl);
	fklFl.close()
	
	fklFl = open("output/testBlocked.pkl", "wb")
	pickle.dump(outputBlocked, fklFl);
	fklFl.close()