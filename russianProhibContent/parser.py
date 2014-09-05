from nltk import SnowballStemmer
import csv
from nltk.tokenize import RegexpTokenizer
import pickle

# parses russian text
# transliterates the text russian text to ascii, stems and stops producing intermediate files (unblocked_train,
# blocked_train, and test files).
# The script expects data/train.tsv data/test.tsv and data/stop_ru.txt(https://github.com/cloudera/search/blob/master/samples/solr-nrt/collection1/conf/lang/stopwords_ru.txt and
# https://sites.google.com/site/kevinbouge/stopwords-lists)

def populate_russian_stopwords():
    with open('data/stopwords_ru.txt', 'rb') as f:
        lines = f.readlines()
        arr = [word.strip().decode('utf-8') for word in lines]

    f.close()
    return arr

stopwords = populate_russian_stopwords()
stemmer = SnowballStemmer('russian')


def transliterate_word(w):
    cyrillic_to_ascii = {
        u'\u0410': 'A', u'\u0430': 'a',
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
        if char in cyrillic_to_ascii:
            converted_word += cyrillic_to_ascii[char]
        else:
            converted_word += char

    return converted_word


def stem_word(word, should_stem):
    if should_stem:
        return stemmer.stem(word)
    return word


def stop_word(word, should_stop):
    if should_stop:
        if word in stopwords:
            return ""

    return word


def normalize_words(words, should_stem, should_stop):
    ret_str = ""
    separator = ""
    for word in words:
        word = stop_word(word, should_stop)
        word = stem_word(word, should_stem)

        if word.strip() != "":
            ret_str += transliterate_word(word.strip()) + separator
            separator = " "

    return ret_str


def load_data(file_name, columnsToUse):

    cleaned_data_blocked = []
    cleaned_data_not_blocked = []

    tokenizer = RegexpTokenizer(r'\w+')

    lines_counter = 0
    with open(file_name,'rb') as fl:
        item_reader = csv.DictReader(fl, delimiter='\t', quotechar='"')

        for i, item in enumerate(item_reader):
            # skip header
            if lines_counter >= 0:
                item = {featureName: featureValue.decode('utf-8') for featureName,featureValue in item.iteritems()}
                for columnToUse in columnsToUse:

                    if columnToUse == 'attrs':
                        ## convert attributes from json to key value pairs, split on comma for kv pairs
                        ## the pairs themselves are split into k=> by colon
                        arr = {}
                        a = item[columnToUse].split('", "')
                        for e in a:
                            p = e.split('":"')
                            if len(p) == 2:
                                arr[normalize_words([p[0]], True, True)] = normalize_words([p[1]], True, True)
                        item[columnToUse] = arr
                    else:
                        words = tokenizer.tokenize(item[columnToUse])
                        item[columnToUse] = normalize_words(words, True, True)

                # do a lazy thing here and split training data into blocked and unblocked
                # all test data will go in the unblocked array
                if 'is_blocked' in item and item['is_blocked'] == u'1':
                    cleaned_data_blocked.append(item)
                else:
                    cleaned_data_not_blocked.append(item)

            lines_counter += 1
            if (lines_counter % 10000) == 0:
                print "Processed ", lines_counter, " from ", file_name

    fl.close()

    return cleaned_data_not_blocked,cleaned_data_blocked



if __name__=="__main__":
    # pickle is slow has hell
    outputUnBlocked, outputBlocked = load_data("data/train.tsv", ['category', 'subcategory', 'title', 'description', 'attrs'])

    fklFl = open("output/trainunblocked.pkl", "wb")
    pickle.dump(outputUnBlocked, fklFl);
    fklFl.close()

    fklFl = open("output/trainblocked.pkl", "wb")
    pickle.dump(outputBlocked, fklFl);
    fklFl.close()

    outputUnBlocked, outputBlocked = load_data("data/test.tsv", ['category', 'subcategory', 'title', 'description', 'attrs'])
    fklFl = open("output/testUnblocked.pkl", "wb")
    pickle.dump(outputUnBlocked, fklFl);
    fklFl.close()

    fklFl = open("output/testBlocked.pkl", "wb")
    pickle.dump(outputBlocked, fklFl);
    fklFl.close()