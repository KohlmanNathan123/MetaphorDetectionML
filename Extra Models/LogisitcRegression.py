#Imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.linear_model import LogisticRegression
import processing


processing.process_data('train.csv','processed_train.csv')
#Load training and test data
data = pd.read_csv('processed_train.csv')
vectorizer = TfidfVectorizer()
X = data['text'].values
Y = data['label'].values

#Format data
X = vectorizer.fit_transform(X).toarray()

#Train/test split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Train
logreg = LogisticRegression(random_state=16)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

#Calculate accuracy
accuracy = logreg.score(X_test,y_test)
print("Accuracy: ", accuracy)
print(classification_report(y_pred, y_test))