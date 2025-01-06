#Imports
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pandas as pd
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

#Train model
model = GaussianNB()
model.fit(X_train.toarray(), y_train)

y_pred = model.predict(X_test.toarray())

#Calculate accuracy
accuracy = model.score(X_test.toarray(),y_test)
print("Accuracy: ", accuracy)

print(classification_report(y_pred, y_test))