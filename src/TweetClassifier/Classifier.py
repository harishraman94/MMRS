import csv
import pickle
import nltk
from SWIMMRS.src.TweetClassifier import Tokenizer, FeatureExtraction
import re


# Read the tweets one by one and process it
negtn_regex = re.compile( r"""(?:^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)$)|n't""", re.X)

def featureListCorpus():
    inpTweets = csv.reader(open('../src/TweetClassifier/sampleDataset.csv', 'r'))
    tweets = []
    for row in inpTweets:
        sentiment = row[2]
        tweet = row[3]
        processedTweet = Tokenizer.preprocess(tweet)
        featureVector = FeatureExtraction.getfeatureVector(processedTweet)
        tweets.append((featureVector, sentiment))
    #print(tweets)
    return tweets

def testData():
    testTweets = csv.reader(open('../src/TweetClassifier/isear1.csv', 'r'))
    tweets = []
    for row in testTweets:
        tweet = row[3]
        processedTweet = Tokenizer.preprocess(tweet)
        featureVector = FeatureExtraction.getfeatureVector(processedTweet)
    return featureVector

def extractFeatures(tweet):
    '''featureList = []
    tweets = featureListCorpus()
    for t in tweets:
        for i in t[0]:
            featureList.append(i)
    featureList = list(set(featureList))
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s) ' % word] = (word in tweet_words)
    return features'''

    bag = {}
    words_uni, words_bi, words_tri = [], [], []
    for words in tweet:
        #words_uni = ['has(%s)' % ug for ug in words]
        words_bi = ['has(%s)' % ','.join(map(str, bg)) for bg in nltk.bigrams(words)]
        words_tri = ['has(%s)' % ','.join(map(str, tg)) for tg in nltk.trigrams(words)]
    for f in words_uni + words_bi + words_tri:
        bag[f] = 1

    # bag = collections.Counter(words_uni+words_bi+words_tri)
    return bag

def extract_features(words):
    features = {}
    word_features = extractFeatures(words)
    features.update(word_features)

    #negation_features = get_negation_features(words)
    #features.update(negation_features)

    return features

def get_negation_features(words):
    INF = 0.0
    for w in words:
        negtn = [bool(negtn_regex.search(w))]
        left = [0.0] * len(words)
        prev = 0.0
        for i in range(0, len(words)):
            if (negtn[i]):
                prev = 1.0
            left[i] = prev
            prev = max(0.0, prev - 0.1)

        right = [0.0] * len(words)
        prev = 0.0
        for i in reversed(range(0, len(words))):
            if (negtn[i]):
                prev = 1.0
            right[i] = prev
            prev = max(0.0, prev - 0.1)

    return dict(zip(
        ['neg_l(' + w + ')' for w in words] + ['neg_r(' + w + ')' for w in words],
        left + right))


def saveClassifier(classifier):
    f = open('../src/TweetClassifier/myClassifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()

def loadClassifier():
    f = open('../src/TweetClassifier/myClassifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier

