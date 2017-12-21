import csv
import pickle

def featureExtraction():
    inpData = csv.reader(open('../src/MusicRecommender/SampleCorpus.csv','r'))
    mood = {}
    sentiment = []
    for row in inpData:
        genre = row[1]
        sentiment.append((row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10]))
        mood[genre] = sentiment
    return mood

def saveClassifier(classifier):
    f = open('../src/MusicRecommender/musicGenreClassifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()

def loadClassifier():
    f = open('../src/MusicRecommender/musicGenreClassifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier
