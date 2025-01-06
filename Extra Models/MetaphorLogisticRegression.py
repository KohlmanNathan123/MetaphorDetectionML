#Imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix

#Load training and test data
data = pd.read_csv('train.csv')

vectorizer = TfidfVectorizer()
X_text_features = vectorizer.fit_transform(data['text'])

numerical_features = data[['metaphorID']].values
numerical_features_sparse = csr_matrix(numerical_features)

X = hstack([X_text_features,numerical_features_sparse])
Y = data['label'].values

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
