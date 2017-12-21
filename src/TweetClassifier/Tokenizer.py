# Text pre-processing
import json
import nltk
import re  # used for tockenizing the tweet

regex_str = [
    r'<[^>]+>',  # HTML Tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # Hashtags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + '$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    #stemWords = []
    #stemmer = nltk.stem.PorterStemmer()
    #for t in tokens:
    #    stemWords.append(stemmer.stem(t))
    #stemWords.append(stemmer.stem(w) for w in tokens)
    #print(stemWords)
    #emojiPresent = emoticonCheck(tokens)
    #if(len(emojiPresent) != 0):
        #print(emojiPresent)
    #else:
    #biGrams(stemWords)
    return tokens

def biGrams(words):
    bGrams = []
    for item in nltk.bigrams(words):
        bGrams.append(' '.join(item))
    return bGrams

def triGrams(words):
    tGrams = []
    for item in nltk.trigrams(words):
        tGrams.append(' '.join(item))
    return tGrams

def emoticonCheck(tweets):
    emoji_pattern = re.compile('[\U0001F300-\U0001F64F]')
    emojis = emoji_pattern.findall(''.join(tweets))
    return emojis

#sample_tweet = 'Never knew Glorious by Macklemore official music video was this sweet.😭'  # 'RT @imHarishRaman: just an example! :D http://example.com #NLP'
#tokenList = (preprocess(sample_tweet))
