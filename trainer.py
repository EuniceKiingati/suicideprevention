import pickle
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk


from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from usefulMethods import tokenizer

nltk.download('stopwords')


def preprocess_text(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    lowercase_text = re.sub('[\W]+', ' ', text.lower())
    text = lowercase_text+' '.join(emoticons).replace('-', '')
    return text


tqdm.pandas()
df = pd.read_csv('suicidal_data.csv')
df['tweet'] = df['tweet'].progress_apply(preprocess_text)


vect = HashingVectorizer(decode_error='ignore', n_features=2 **
                         21, preprocessor=None, tokenizer=tokenizer)



clf = SGDClassifier(loss='log', random_state=1)

X = df["tweet"].to_list()
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

X_train = vect.transform(X_train)
X_test = vect.transform(X_test)

classes = np.array([0, 1])
clf.partial_fit(X_train, y_train, classes=classes)

print('Accuracy: %.3f' % clf.score(X_test, y_test))
clf = clf.partial_fit(X_test, y_test)
label = {0: 'negative', 1: 'positive'}
example = ["I'll kill myself am tired of living depressed and alone"]
X = vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%' %
      (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))


f = open('classifier.pickle', 'wb')
pickle.dump(clf, f)
f.close

# label = {0: 'negative', 1: 'positive'}
# example = ["It’s such a hot day, I’d like to have ice cream and visit the park"]
# X = vect.transform(example)
# print('Prediction: %s\nProbability: %.2f%%' %
#       (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))

# example = ["I am just done, i think this ist it, I cannot do it anymore, i will end it today"]
# X = vect.transform(example)
# print('Prediction: %s\nProbability: %.2f%%' %
#       (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))

# example = ["What is the point of living, I am done"]
# X = vect.transform(example)
# print('Prediction: %s\nProbability: %.2f%%' %
#       (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))
