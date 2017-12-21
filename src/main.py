import nltk
import sklearn

'''
1 JOY
2 FEAR
3 ANGER 
4 SADNESS
5 DISGUST
6 SHAME
7 GUILT
'''

from SWIMMRS.src.MusicRecommender import GenreClassifier
from SWIMMRS.src.TweetClassifier import Tokenizer, FeatureExtraction, Classifier
from SWIMMRS.src.scraper.Datascraper import DataScraper
from SWIMMRS.src.scraper.OauthConnectionClient import OauthClient

userTweets_url = 'https://api.twitter.com/1.1/statuses/user_timeline.json?user_id=2597834149&count=20'
userProfile_url = 'https://api.twitter.com/1.1/users/show.json?user_id=2597834149'

def main():
    oauth_obj = OauthClient()
    oauth_client = oauth_obj.getOauthClient()

    data_scraper = DataScraper(oauth_client)
    user_tweets = data_scraper.getUserTweets(userTweets_url)
    profileInfo = data_scraper.getUserProfileData(userProfile_url)
    #print(profileInfo['name'], profileInfo['location'])

    #for a in user_tweets:
    #    tokenList = Tokenizer.preprocess(a['text'])
    #    featureList = FeatureExtraction.getfeatureVector(tokenList)

    #tweets = Classifier.featureListCorpus()
    #trainingSet = nltk.classify.apply_features(Classifier.extract_features, tweets)
    #NBClassifier = nltk.NaiveBayesClassifier.train(trainingSet)

    #Classifier.saveClassifier(NBClassifier)
    cls = Classifier.loadClassifier()
    #testTweet = 'I bought a phone.'
    #processedTestTweet = Tokenizer.preprocess(testTweet)
    testFeatureVector = Classifier.testData()
    #print(cls.classify(testFeatureVector))
    print(nltk.classify.accuracy(cls, testFeatureVector))
    #print(cls.show_most_informative_features(10))

    #mood = GenreClassifier.featureExtraction()
    #print(mood)

if __name__ == '__main__':
    main()

