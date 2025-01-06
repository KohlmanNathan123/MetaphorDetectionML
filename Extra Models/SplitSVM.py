#Imports
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from scipy.sparse import hstack, csr_matrix
import pandas as pd

#Load training and test data
data = pd.read_csv('train-1.csv')

vectorizer = TfidfVectorizer()

X =  vectorizer.fit_transform(data['text'].values).toarray()
Y = data['label'].values

totalPred = []
totalTest = []

#Split data into categories and train seperately
categories = data['metaphorID'].values
for j in range(7):
    catX = []
    catY = []
    for i in range(len(X)):
        if(categories[i] == j):
            catX.append(X[i])
            catY.append(Y[i])

    #Train/test split 80/20
    X_train, X_test, y_train, y_test = train_test_split(catX, catY, test_size=0.2, random_state=0)

    #Train model
    model = svm.SVC(kernel="sigmoid", class_weight={0:3, 1:1})
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    
    #append predictions to total prediction array
    for k in range(len(pred)):
        totalPred.append(bool(pred[k]))
        totalTest.append(bool(y_test[k]))

#Final report with combined predictions
correct = 0
for i in range(len(totalPred)):
    if(totalPred[i] == totalTest[i]):
        correct+=1

print(f"Accuracy: {correct/len(totalPred)}")

print(classification_report(totalPred, totalTest))