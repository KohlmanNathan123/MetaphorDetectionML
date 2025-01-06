import nltk 
import pandas as pd
import csv

#Read data
data = pd.read_csv('train-1.csv')
text = data['text'].values
metaphorIDs = data['metaphorID']

#Part of speech tagger using NLTK
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')

#List of potential words
metaphorWords = ["road", "candle", "light", "spice", "ride", "train", "boat"]

partsOfSpeech = []
for i in range(len(text)):
    #tokenize words in text
    tokenized = nltk.word_tokenize(text[i])
    tokenized = nltk.pos_tag(tokenized)
    metaphorW = metaphorWords[metaphorIDs[i]]

    notFound = True
    j = 0
    while notFound:
        if metaphorW in tokenized[j][0].lower():
            partsOfSpeech.append(tokenized[j][1])
            notFound = False
        j+=1

#print(partsOfSpeech)

with open('parts-of-speech.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in partsOfSpeech:
        writer.writerow([row])