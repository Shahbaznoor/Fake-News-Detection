import pickle
import re
import string

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df_fake = pd.read_csv("data/Fake.csv")
df_true = pd.read_csv("data/True.csv")

# 1 for true class and 0 for fake class
df_fake["class"] = 0
df_true["class"] = 1

df_merge = pd.concat([df_fake, df_true], axis =0 )

df = df_merge.drop(["title", "subject","date"], axis = 1)

df = df.sample(frac = 1)

df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)

def wordopt(text):
    text = text.lower()
    text = re.sub('\\[.*?\\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    return text

df["text"] = df["text"].apply(wordopt)

x = df["text"]
y = df["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

vectorization = TfidfVectorizer(max_features=10000)
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

pred_dt = DT.predict(xv_test)

print(DT.score(xv_test, y_test))

print(classification_report(y_test, pred_dt))

file = open('model/dt_model_Fake', 'wb')

saved_rf_model_AI = pickle.dump(DT, file=file)

with open('model/Fake_tfidf_vectorizer', 'wb') as f:
    pickle.dump(vectorization, f)