#Imports
from sklearn import svm
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

#Load training and test data
data = pd.read_csv('train-1-processed.csv')

vectorizer = TfidfVectorizer()

X =  vectorizer.fit_transform(data['text'].values).toarray()
Y = data['label'].values

#Train/test split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

model = svm.SVC(kernel="sigmoid")
model.fit(X_train,y_train)

pred = model.predict(X_test)

#Calculate accuracy
accuracy = model.score(X_test,y_test)
print("Accuracy: ", accuracy)

print(classification_report(pred, y_test))
