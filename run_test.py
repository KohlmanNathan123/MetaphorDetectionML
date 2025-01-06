# imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pandas as pd
import processing 
import pickle

# pre process the data
processing.process_data('train.csv', 'processed_train.csv')
processing.process_data('test.csv', 'processed_test.csv')

# load train and test data
data = pd.read_csv('processed_train.csv')
test = pd.read_csv('processed_test.csv')

# initialize vectorizer
vectorizer = TfidfVectorizer()

# format training and testing data
X =  vectorizer.fit_transform(data['text'].values).toarray()
X_test = vectorizer.transform(test['text'].values).toarray()
Y_test = test['label'].values

totalPred = []
totalTest = []

# read serialized models into test program
with open('models.pkl', 'rb') as file:
    models = pickle.load(file)

# split data into categories and test seperately
test_categories = test['metaphorID'].values
for j in range(7):
    # sort test data into metaphor ID categories
    tX = []
    tY = []
    for i in range(len(X_test)):
        if(test_categories[i] == j):
            tX.append(X_test[i])
            tY.append(Y_test[i])

    if(len(tX) != 0):
        pred = models[j].predict(tX)
    
        # append predictions to total prediction array
        for k in range(len(pred)):
            totalPred.append(bool(pred[k]))
            totalTest.append(bool(tY[k]))

# final report with combined predictions
correct = 0
for i in range(len(totalPred)):
    if(totalPred[i] == totalTest[i]):
        correct+=1


print(f"Accuracy: {correct/len(totalPred)}")
print(classification_report(totalPred, totalTest))

# write predictions to csv file
with open("predictions.csv", "w", newline="") as file:
    for row in totalPred:
        file.write(f"{str(row)}\n")