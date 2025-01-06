# imports
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pandas as pd
import processing
import pickle

# pre processs the training data
processing.process_data('train.csv', 'processed_train.csv')

# load training  data
data = pd.read_csv('processed_train.csv')

# initialize vectorizer
vectorizer = TfidfVectorizer()

# add text column as a feature value
X =  vectorizer.fit_transform(data['text'].values).toarray()
# add labels
Y = data['label'].values

# combined final predictions
totalPred = []
# stores all 7 models
models = []

# split data into categories and train seperate models
categories = data['metaphorID'].values
for j in range(7):
    # sort training data into metaphor ID categories
    catX = []
    catY = []
    for i in range(len(X)):
        if(categories[i] == j):
            catX.append(X[i])
            catY.append(Y[i])

    # train/test variables
    X_train = catX
    y_train = catY

    # train model with SVM
    model = svm.SVC(random_state=0, class_weight={0:3, 1:1})
    model.fit(X_train, y_train)
    # add each model to an array of models
    models.append(model)

# serialize the models
with open('models.pkl', 'wb') as file:
    pickle.dump(models, file)