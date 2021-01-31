#!/usr/bin/python3

def message_rating(message):
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB


    df = pd.read_csv('spam.csv')

    messages = pd.DataFrame(df, columns=['rating', 'message'])

    message_train = messages['message']
    rating_train = messages['rating']

    model = CountVectorizer()
    model.fit(message_train)
    #create a dictionary with value of position of word

    transformed_message_train = model.transform(message_train)
    #create a matrix with frequency of a word and the word

    classifier = MultinomialNB()
    classifier.fit(transformed_message_train, rating_train)

    test_counts = model.transform([message])

    final = classifier.predict(test_counts)
    chances = round((float(classifier.predict_proba(test_counts)[0][1]*100)),5)

    #print(final,type(final),chances,type(chances))

#Prediction is numpy array, another conditionnal to return a string

    if final == 'spam':
        return_statement = 'This message is spam' + '\n' + 'This message has a probability of {0}% of being spam'.format(chances)
    else:
        return_statement = 'This message is not spam' + '\n' + 'This message has a probability of {0}% of being spam'.format(chances)

#I removed the spam update program. I'll put later to see if it works on Android:
#If we export as APK, then we must export the database of spam messages for Bayes.

    return str(return_statement)

