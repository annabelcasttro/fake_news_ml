import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
import string
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

# Reading true news
df_truenews = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")
df_truenews.head()

# Reading fake news
df_fakenews = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")
df_fakenews.head()

# Adding label column
df_fakenews['label'] = 'false'
df_truenews['label'] = 'true'
# Amount of fake news
fake_size=len(df_fakenews)
print("Amount of fake news:", fake_size)
# Amount of true news
true_size=len(df_truenews)
print("Amount of true news:", true_size)

# fake news dataset
df_fakenews.info()
print(df_fakenews.describe(include='all'))

# true news dataset
df_truenews.info()
print(df_truenews.describe(include='all'))

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
print(df_fakenews.date.value_counts())

# Amount of wrong date info (urls rather than date) in fake news
len(df_fakenews.text[df_fakenews['date'].str.contains('https')])

# Amount of wrong date info (urls rather than date) in true news
len(df_truenews.text[df_truenews['date'].str.contains('https')])

print(df_fakenews.subject.value_counts())
print(df_truenews.subject.value_counts())

missing_values=[' ', 'na']
df_fakenews=df_fakenews.replace(missing_values,np.NaN)
df_truenews=df_truenews.replace(missing_values,np.NaN)
print("NA - True News:\n", df_truenews.isna().sum())
print("Null - True News:\n", df_truenews.isnull().sum())
print("NA - Fake News:\n", df_fakenews.isna().sum())
print("Null - Fake News:\n", df_fakenews.isnull().sum())

# Verifying duplicates in the text column
values=df_fakenews.text.value_counts()
print(sum(values>1))

# Verifying unique duplicates in the text column
values=df_truenews.text.value_counts()
print(sum(values>1))

# Printing an example of duplicated news
print(df_truenews[df_truenews.text==(values[values>1]).keys()[0]])

# Showing the amount of duplicated lines, comparing title, text and date. Only subject has changed.
count=len(df_fakenews[df_fakenews.duplicated(subset=["title", "text", "subject", "date"])])
print("Amount of duplicated lines (title, text, subject and date): ", count, " ", round(100*count/fake_size,2), "%")
# When removing subject...
count=len(df_fakenews[df_fakenews.duplicated(subset=["title", "text", "date"])])
print("Amount of duplicated lines (title, text and date): ", count, " ", round(100*count/fake_size,2), "%")
# When removing date, we can see more 2 examples.
count=len(df_fakenews[df_fakenews.duplicated(subset=["title", "text"])])
print("Amount of duplicated lines (title, text): ",  count, " ", round(100*count/fake_size,2), "%")
# When removing title, we can see more examples.
count=len(df_fakenews[df_fakenews.duplicated(subset=["text"])])
print("Amount of duplicated lines (text): ",  count, " ", round(100*count/fake_size,2), "%")

# Showing the amount of duplicated lines, comparing title, text and date. Only subject has changed.
count=len(df_truenews[df_truenews.duplicated(subset=["title", "text", "subject", "date"])])
print("Amount of duplicated lines (title, text, subject and date): ", count, " ", round(100*count/true_size,2), "%")
# When removing subject...
count=len(df_truenews[df_truenews.duplicated(subset=["title", "text", "date"])])
print("Amount of duplicated lines (title, text and date): ", count, " ", round(100*count/true_size,2), "%")
# When removing date, we can see more 2 examples.
count=len(df_truenews[df_truenews.duplicated(subset=["title", "text"])])
print("Amount of duplicated lines (title, text): ",  count, " ", round(100*count/true_size,2), "%")
# When removing title, we can see more examples.
count=len(df_truenews[df_truenews.duplicated(subset=["text"])])
print("Amount of duplicated lines (text): ",  count, " ", round(100*count/true_size,2), "%")


# Merging datasets
df_news = pd.concat([df_truenews, df_fakenews], ignore_index=True)
df_news.head()

# Info
df_news.info(verbose = True)

# Describe
df_news.describe(include = 'all')

# Some values aren't date
df_news[df_news.date.isna()]

pd.set_option('display.max_rows', 500)

# Fixing the date type
df_news['date'] = pd.to_datetime(df_news['date'], errors = 'coerce')

print("Amount of duplicated lines: ", sum(df_news.duplicated(subset=["subject", "title", "text", "date"])))
print("Amount of null values:\n", df_news.isnull().sum())

# Removing empty texts
df_news.dropna(inplace=True)
df_news.reset_index(drop=True, inplace=True)

# Dropping duplicated lines
df_news.drop_duplicates(inplace=True)
df_news.reset_index(drop=True, inplace=True)

# Validation
print("Amount of duplicated lines: ", sum(df_news.duplicated(subset=["subject", "title", "text", "date"])))
print("Amount of null values:\n", df_news.isnull().sum())

## GRAPHICS

# News X Subjects
test = df_news.groupby(['subject', 'label']).size()
test

test.plot(kind='bar')

# Checking if the labels are balanced
df_news.groupby(['label']).size().plot(kind='bar')

## Text Cleaning

# transforming text to lower case
df_news.text = df_news.text.apply(lambda x: x.lower())
df_news.text[0]


t=df_news.text.str.contains('reuters')
print(round(t.sum()/len(df_news)*100,2), "% having reuters text")
print(len(t))
round(df_news[t==True].groupby('label').size()/len(df_news[t==True])*100, 2)

# analyzing the first word (mostly from washington)
t=df_news.text.str.split().str.get(0)
t.value_counts().nlargest(20).plot(kind='bar')
print(t.value_counts().nlargest(20).index)

# analyzes by label
df_news['first'] = t
k=df_news.groupby(['first', 'label']).size()
k.nlargest(20).plot(kind='barh')

t=df_news.text.str.contains('washington')
print(round(t.sum()/len(df_news)*100,2), "% having washington text")
print(len(t))
round(df_news[t==True].groupby('label').size()/len(df_news[t==True])*100, 2)

df_news.text = df_news.text.apply(lambda x: re.sub('reuters', ' ', x))
df_news.text = df_news.text.apply(lambda x: re.sub('washington', ' ', x))
df_news.text = df_news.text.apply(lambda x: re.sub('\(', ' ', x))
df_news.text = df_news.text.apply(lambda x: re.sub('\)', ' ', x))
df_news.text = df_news.text.apply(lambda x: re.sub('\-', ' ', x))

# Removing mentions (@)
df_news.text = df_news.text.apply(lambda x: re.sub('@\S+', ' ', x))
df_news.text[2764]

# Removing monetary symbols ($)
df_news.text = df_news.text.apply(lambda x: re.sub('\$', ' ', x))

# Removing digits
df_news.text = df_news.text.apply(lambda x: re.sub('\w*\d\w*', ' ', x))

# Removing URLs
df_news.text = df_news.text.apply(lambda x: re.sub('https?://\S+|www\.\S+', ' ', x))

# Removing URL twitter
df_news.text = df_news.text.apply(lambda x: re.sub(r'pic.twitter.com/[\w]*', ' ', x))

# Removing bit.ly/
df_news.text = df_news.text.apply(lambda x: re.sub('bit.ly/', ' ', x))

# Removing \xa0
df_news.text = df_news.text.apply(lambda x: re.sub(u'\xa0', ' ', x))

# Removing punctuation
pont = set(string.punctuation)
df_news.text = df_news.text.apply(lambda x: "".join([ch for ch in x if ch not in pont]))

# Removing ""' and spaces
df_news.text = df_news.text.apply(lambda x: re.sub('”', ' ', x))
df_news.text = df_news.text.apply(lambda x: re.sub(' +', ' ', x))

# Stemming
stemmer = SnowballStemmer("english")
df_news['SplittedText'] = df_news['text'].str.split()
df_news['SteammedText'] = df_news['SplittedText'].apply(lambda x: [stemmer.stem(y) for y in x])
df_news.head()

stop = stopwords.words('english')
print(stop)

df_news['NoStopWords'] = df_news['SteammedText'].apply(lambda x: ' '.join([item for item in x if item not in stop]))
df_news.head()

## MODEL

# Transforming labels in integers
df_news.loc[df_news['label']=='true', 'label'] = 1
df_news.loc[df_news['label']=='false', 'label'] = 0
df_news.head()

df_news.label.value_counts()

x_train, x_test, y_train, y_test = train_test_split(df_news['NoStopWords'], df_news['label'], test_size = 0.30, random_state = 0)
print(f' * x_train: {x_train.shape}')
print(f' * y_train: {y_train.shape}')

print(f' * x_test: {x_test.shape}')
print(f' * y_test: {y_test.shape}')

y_train=y_train.astype('int')
y_test=y_test.astype('int')

count_nb = Pipeline([('vec', CountVectorizer()), ('clf_multi_nb', MultinomialNB(alpha = 0.001))])
count_svm = Pipeline([('vec', CountVectorizer()), ('clf_svm_linear', LinearSVC(C = 1, max_iter = 100000))])
tfidf_nb = Pipeline([('tfidf', TfidfVectorizer()), ('clf_multi_nb', MultinomialNB(alpha = 0.001))])
tfidf_svm = Pipeline([('tfidf', TfidfVectorizer()), ('clf_svm_linear', LinearSVC(C = 3.0))])
hash_nb = Pipeline([('hash', HashingVectorizer(alternate_sign=False)), ('clf_multi_nb', MultinomialNB(alpha = 0.001))])
hash_svm = Pipeline([('hash', HashingVectorizer(alternate_sign=False)), ('clf_svm_nusvc', NuSVC())])


accuracies = {}

count_nb.fit(x_train.values, y_train.values)
y_pred_count_nb = count_nb.predict(x_test.values)
accuracies['countVectorizer_multinomialNB'] = accuracy_score(y_pred_count_nb, y_test)

count_svm.fit(x_train.values, y_train.values)
y_pred_count_svm = count_svm.predict(x_test.values)
accuracies['countVectorizer_linearSVC'] = accuracy_score(y_pred_count_svm, y_test)

tfidf_nb.fit(x_train.values, y_train.values)
y_pred_tfidf_nb = tfidf_nb.predict(x_test.values)
accuracies['tfidfVectorizer_multinomialNB'] = accuracy_score(y_pred_tfidf_nb, y_test)

tfidf_svm.fit(x_train.values, y_train.values)
y_pred_tfidf_svm = tfidf_svm.predict(x_test.values)
accuracies['tfidfVectorizer_linearSVC'] = accuracy_score(y_pred_tfidf_svm, y_test)

hash_nb.fit(x_train.values, y_train)
y_pred_hash_nb = hash_nb.predict(x_test.values)
accuracies['hashVectorizer_multinomialNB'] = accuracy_score(y_pred_hash_nb, y_test)

hash_svm.fit(x_train.values, y_train)
y_pred_hash_svm = hash_svm.predict(x_test.values)
accuracies['hashVectorizer_linearSVC'] = accuracy_score(y_pred_hash_svm, y_test)
print(accuracies)

plt.figure(figsize = (13, 8))
bar = plt.bar(accuracies.keys(), accuracies.values())
for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2., 1.0001*height,
            '%.2f' % (height*100) + "%", ha = 'center', va = 'bottom', fontweight = 'bold', fontdict = dict(fontsize = 12))
plt.title("Pipelines Results", weight = 'bold', size = 16)
plt.xlabel("Pipelines", weight = 'bold', size = 12)
plt.ylabel("Score", weight = 'bold', size = 12)
plt.xticks(rotation = 90)
plt.show();
