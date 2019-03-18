import pandas as pd
import numpy as np

import utils
import importlib
importlib.reload(utils)
from utils import *


import re, string, collections, bcolz, pickle, os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import confusion_matrix

'''
Train a sentiment analysis classifier given the 
games reviews in the dataset. The model must classify text reviews in positive or negative
'''

df = pd.read_csv('bayesian/cleaned_data/cleanedreviews.csv', lineterminator='\n')
print(df.shape)
df = df[['text', 'score']]
print(df.head())

'''
The data have 127699 reviews but we will keep just the most polarized ones. 
The filter is the next: 
Positive: score > 85 
Negative: score < 50
'''


print(f'reviews with score greater than 85: {len(df.loc[df["score"] > 85])}')
print(f'reviews with score less than 50: {len(df.loc[df["score"] < 50])}')



pos = df.loc[df['score'] > 85, 'text'].copy().reset_index(drop=True)
neg = df.loc[df['score'] < 50, 'text'].copy().reset_index(drop=True)

print(len(pos), len(neg))
# Let's print some positive reviews examples
for i in range(4):
    print(''.join(pos[np.random.randint(0, len(pos))]))
    print('\n')

# Let's print some negative reviews examples
    
for i in range(4):
    print(''.join(neg[np.random.randint(0, len(neg))]))
    print('\n')
    
    
# add the labels: 0 for negative reviews, 1 for positive reviews
    
    
neg = pd.concat([pd.DataFrame(neg), pd.DataFrame(np.zeros(neg.shape), columns=['class'])], 1)
pos = pd.concat([pd.DataFrame(pos), pd.DataFrame(np.ones(pos.shape), columns=['class'])], 1)
print(neg.head())

# Mean, standard deviation and max length of negative reviews

lens = neg['text'].str.len()
print(lens.mean(), lens.std(), lens.max())

lens.hist(figsize=(12, 6), bins=25);


# Reviews with more than 5000 characters are dropped
long_reviews = neg.loc[neg['text'].str.len() > 5000].index
neg.drop(long_reviews, inplace=True)

# Mean, standard deviation and max length of positive reviews

lens = pos['text'].str.len()
lens.mean(), lens.std(), lens.max()

lens.hist(figsize=(12, 6), bins=25);


long_reviews = pos.loc[pos['text'].str.len() > 5000].index
pos.drop(long_reviews, inplace=True)

'''
Is desirable to have a balanced dataset (similar quantity of positive and 
negative instances). So we will pick a random subset of the positive instances
'''

np.random.seed(42)
rand = np.random.permutation(pos.shape[0])
pos = pos.iloc[rand[:neg.shape[0]]].reset_index(drop=True)

print(neg.shape, pos.shape)

# concatenate positive and negative reviews

df = pd.concat([pos, neg]).sample(frac=1).reset_index(drop=True)
print(df.head())

print(df.shape)

# Split data into train and test set

X_train, X_test, y_train, y_test = train_test_split(df['text'].values, df['class'].values, test_size=0.2, random_state=42)

print(len(X_train), len(X_test), len(y_train), len(y_test))

# text mining
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): 
    return re_tok.sub(r' \1 ', s).split()
# Creating bag of words
vect = CountVectorizer(tokenizer=tokenize)

tf_train = vect.fit_transform(X_train)
tf_test = vect.transform(X_test)

'''
Train term frequency contains 32675 documents and 45424 tokens. 
Each row represents a document and each column how many times 
that token appears on the document.
'''

vocab = vect.get_feature_names()
print(len(vocab))

print(vocab)
print(X_train[0])

w0 = set([o for o in X_train[0].split(' ')])

print(w0)


print(vect.vocabulary_['unless'])

print(tf_train[0, 41989])

# Naive Bayes

svd = TruncatedSVD()
reduced_tf_train = svd.fit_transform(tf_train)
plot_embeddings(reduced_tf_train, y_train)

p = tf_train[y_train==1].sum(0) + 1
q = tf_train[y_train==0].sum(0) + 1
r = np.log((p/p.sum())/(q/q.sum()))
b = np.log(len(p)/len(q))

pre_preds = tf_test @ r.T + b
preds = pre_preds.T > 0
acc = (preds==y_test).mean()
print(f'Accuracy: {acc}')

model = LogisticRegression(C=0.2, dual=True)
model.fit(tf_train, y_train)
preds = model.predict(tf_test)
acc = (preds==y_test).mean()
print(f'Accuracy: {acc}')

plot_confusion_matrix(confusion_matrix(y_test, preds.T), classes=['Negative', 'Positive'], title='Confusion matrix')


coef_df = pd.DataFrame({'vocab': vocab, 'coef':model.coef_.reshape(-1)})
pos_top10 = coef_df.sort_values('coef', ascending=False).reset_index(drop=True)[:10]
neg_top10 = coef_df.sort_values('coef').reset_index(drop=True)[:10]



fig, axs = plt.subplots(1, 2, figsize=(8, 8))
fig.subplots_adjust(wspace=0.8)
pos_top10.sort_values('coef').plot.barh(legend=False, ax=axs[0])
axs[0].set_yticklabels(pos_top10['vocab'].values.tolist()[::-1])
axs[0].set_title('Positive');
neg_top10.sort_values('coef', ascending=False).plot.barh(legend=False, ax=axs[1])
axs[1].set_yticklabels(neg_top10['vocab'].values.tolist()[::-1])
axs[1].set_title('Negative');

vect = TfidfVectorizer(strip_accents='unicode', tokenizer=tokenize, ngram_range=(1, 2), max_df=0.8, min_df=4, sublinear_tf=True)


tfidf_train = vect.fit_transform(X_train)
tfidf_test = vect.transform(X_test)


svd = TruncatedSVD()
reduced_tfidf_train = svd.fit_transform(tfidf_train)



plot_embeddings(reduced_tfidf_train, y_train, 2000)

model = LogisticRegression(C=0.2, dual=True)
model.fit(tfidf_train, y_train)
preds = model.predict(tfidf_test)
acc = (preds==y_test).mean()
print(f'Accuracy: {acc}')

plot_confusion_matrix(confusion_matrix(y_test, preds.T), classes=['Negative', 'Positive'], title='Confusion matrix')




























