# imports
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pandas as pd

# load training data
data = pd.read_csv('train.csv')

# initialize vectorizer
vectorizer = TfidfVectorizer()

X = data['text'].values
Y = data['label'].values

# format data
X = vectorizer.fit_transform(X).toarray()

# train/test split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# train model
model = GaussianNB()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(pred)

# calculate accuracy
accuracy = model.score(X_test,y_test)

# print results
print("Accuracy: ", accuracy)
print(classification_report(pred, y_test))