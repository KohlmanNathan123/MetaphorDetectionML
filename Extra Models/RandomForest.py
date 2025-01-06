#Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from scipy.sparse import hstack, csr_matrix
import pandas as pd

#Load training and test data
data = pd.read_csv('train-1-processed.csv')

#read split data
with open("split-sentences.csv", "r") as file:
    split_data = file.readlines()

vectorizer = TfidfVectorizer()

text = data['text'].values
Y = data['label'].values
X_split = vectorizer.fit_transform(split_data).toarray()

#Format data
vectorized_text = vectorizer.fit_transform(text).toarray()

#Add metaphor ID as feature in model
ids = data['metaphorID'].values
X = vectorized_text

#Split data into categories
categories = data['metaphorID'].values
catX = []
catY = []
for i in range(len(X)):
    if(categories[i] == 0):
        catX.append(X[i])
        catY.append(Y[i])

#Train/test split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

#Train model
model = RandomForestClassifier(max_depth=100, class_weight={0:3, 1:1})
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(pred)

#Calculate accuracy
accuracy = model.score(X_test,y_test)
print("Accuracy: ", accuracy)

print(classification_report(pred, y_test))