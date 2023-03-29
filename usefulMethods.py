import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer

porter = PorterStemmer()
stop = stopwords.words('english')

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower())
    text += ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in tokenizer_porter(text) if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore', n_features=2 **
                         21, preprocessor=None, tokenizer=tokenizer)
