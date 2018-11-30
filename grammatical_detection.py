#importing dependencies ..

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import linear_model
import pandas as pd
import numpy as np
import re

english_profile = pd.read_csv('../English Profile/aaa_clean_sent.csv', sep='\t',names=['Sentence','Level'])

#if we want to use GloVe Embeddings
def vectorize(glove_loc):
	word_embed_dict = dict()
	with open(glove_loc,"r") as f:
	    for line in f.readlines():
	        word_embed_dict[line.split()[0]]= np.float32(line.strip().split()[1:])
	return word_embed_dict

def sentence_embed(sentence):
    # Create Sentence embeddings
    words = [i.lower() for i in re.sub("[^a-zA-Z ]+", "", str(sentence)).strip().split(" ") if i!=""]
#    words = [i for i in sentence.split()]
    w_vectors = np.zeros(300)
    for w in words:
        if w in word_embed_dict:
            w_vectors+=np.array(word_embed_dict[w]).astype(np.float)
#    w_vectors = w_vectors / len(words)
    return w_vectors

def posify(sent, flag='mixed'):
    #returns a pos-tagged sequences, or pos-word sequence.

    sent = sent.strip()

    tokenized = nltk.tokenize.word_tokenize(sent)

    pos = nltk.pos_tag(tokenized)

    mixed_list = list()
    for t in pos:
        if t[1] in ['CC','IN','RB']:
            mixed_list.append(t[0])
        else:
            mixed_list.append(t[1])
    if flag == 'words': # returns the sentence intact
        return sent
    if flag == 'POS': # returns POS tagged sequence instead
        return ' '.join([t[1] for t in pos])
    if flag == 'mixed': # a mixed of both
        return ' '.join(mixed_list)


english_profile['tokens'] = english_profile.Sentence.apply(posify, args=('words',))

corpus = english_profile['tokens']


vectorizer = CountVectorizer(lowercase=False, ngram_range=(1, 3))
mat = vectorizer.fit_transform(corpus)

def convert(label):
    dic = {"A1":1,"A2":2,"B1":3,"B2":4,"C1":5,"C2":6}
    return dic[str(label)]
        
english_profile['vectors'] = english_profile.Sentence#.apply(sentence_embed)

english_profile['label'] = english_profile.Level.apply(convert)

# add polynomial features e.g. x+bx^2+cx^3+ ...+ x^n. Warn: Exponential Feature Growning!
poly = PolynomialFeatures(2)
X = pd.DataFrame(english_profile['vectors'].values.tolist())
poly_x = poly.fit_transform(mat)
#y = to_categorical(pd.DataFrame(data['label'].values.tolist()))

X_train, X_test, y_train, y_test = train_test_split(poly_x, english_profile['label'] , test_size=0.33, random_state=42)

mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg',n_jobs=-1, random_state=42)
mul_lr.fit(X_train, y_train)


print("Multinomial Logistic regression Test Accuracy: Train : ", accuracy_score(y_train, mul_lr.predict(X_train)))
print("Multinomial Logistic regression Test Accuracy: Test : ", accuracy_score(y_test, mul_lr.predict(X_test)))
print(classification_report(y_test, mul_lr.predict(X_test)))
print(confusion_matrix(y_test, mul_lr.predict(X_test)))