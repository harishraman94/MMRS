# MMRS
Mood based Music Recommendation System

The goal of the project was to develop a Mood-based Music Recommendation system that is based on context analysis. In this project, the recent user tweets are parsed and a mood corresponding to the parsed tweets are identified. This mood is analyzed further and a corresponding genre of the music is predicted. A song from the found genre is returned back to the user, which corresponds to the sentiment as detected from his recent tweets. The focus of the project is to understand how parsing, sentiment analysis, and Decision tree works, and to build a system using the various concepts present in Language Processing and Machine Learning.

Key fucntionalities implemented in the project are:

1. The tweets were scraped from the user account, by authenticating the application with Twitter using Twitter API and Requests                    library. Once authenticated, user tweets, along with basic user information can be retrieved in JSON format.
2. The tweets are then pre-processed. Here, they undergo pre-process techniques such as Stemming, Tokenization, Hashtag, URL, handles, emoticons identification.
3. Tweets containing emojis are tokenized separately and the emojis are analysed, and a sentiment is attached to it that is being represented by the emoji. This modified tweet is then fed as input to subsequent levels of analysis.
4. Tweets that doesn’t have an emoji undergoes the usual pre-processing. Once tokenized, N-gram models are used to improve the accuracy of context based sentiment identification.
5. The tweets are fed to a Naïve Bayes classifier, which takes in the tokens as input and classifies it into 7 different mood categories. They are Joy, Anger, Sadness, Disgust, Fear, Shame and Guilt.
6. The classified mood, along with the user profile information such as location of the tweet posted, gender of the user (obtained using Gender API) are fed into a Random Forest Classifier, that trains on these features and outputs the genre.
7. Spotify API is then used to fetch a song that is under the outputted genre.

