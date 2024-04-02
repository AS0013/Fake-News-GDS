import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import numpy as np

cleaned_stemmed = pd.read_csv("995,000_cleaned_stemmed2.csv", dtype=str)

# remove the rows with missing values in the 'type' column and in the 'content' column
cleaned_stemmed = cleaned_stemmed.dropna(subset=['type', 'content'])

# replace 'political' and 'clickbait' with 'reliable'
cleaned_stemmed['type'] = cleaned_stemmed['type'].replace(['political', 'clickbait'], 'reliable')

# replace 'bias' and 'satire' with 'fake'
cleaned_stemmed['type'] = cleaned_stemmed['type'].replace(['bias', 'satire'], 'fake')

#  remove all the other types of news except 'reliable' and 'fake'
cleaned_stemmed = cleaned_stemmed[cleaned_stemmed['type'].isin(['reliable', 'fake'])]

print('Total rows after removing missing values:', cleaned_stemmed.shape[0])





CountVectorizer = CountVectorizer()
X = CountVectorizer.fit_transform(cleaned_stemmed['content'])

X_train, X_test_1, y_train, y_test_1  = train_test_split(X, cleaned_stemmed['type'], test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test_1, y_test_1, test_size=0.5, random_state=42)





# Normalizing vectors
from sklearn.preprocessing import StandardScaler

scale = StandardScaler(with_mean=False)

X_train = scale.fit_transform(X_train)
X_val = scale.transform(X_val)





#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)





# Alternative method
text_clf = Pipeline([
    ('vect', CountVectorizer),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])



text_clf.fit(X_train, y_train)
Pipeline(...)
predicted = text_clf.predict(X_val)
np.mean(predicted == y_train)