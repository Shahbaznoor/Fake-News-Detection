import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Import modules from NLTK
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Dataset AI vs Human
df = pd.read_csv("data/Training_Essay_Data.csv")
df.columns = ['text', 'label']

def case_folding(text):
    return text.lower()  # lower all the text

df['case_folded_text'] = df['text'].apply(case_folding)


def clean_text(text):
    text = re.sub(r"http\S+", "", text)    # remove URLs
    text = re.sub(r"@\w+", "", text)       # remove mention
    text = re.sub(r"#\w+", "", text)       # remove hashtag
    text = re.sub(r"[^\w\s]", "", text)    # remove punctuation marks
    text = re.sub(r"\d+", "", text)        # remove numbers
    text = re.sub(r"_", "", text)          # remove underlines
    text = text.strip()                    # remove extra spaces at start/end
    return text

df['clean_text'] = df['case_folded_text'].apply(clean_text)

stop_words = set(stopwords.words('english'))
df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([word
for word in x.split() if word not in stop_words]))


lemmatizer = WordNetLemmatizer()
df['clean_text'] = df['clean_text'].apply(
    lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()])
)


df['tokens'] = df['clean_text'].apply(lambda x: x.split())

print(df[['text', 'clean_text', 'tokens']].head(10))

tfidf = TfidfVectorizer(max_features=10000)
# tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df['clean_text']).toarray()


# View total features (unique words)
unique_words_tfidf = tfidf.get_feature_names_out()
print(f"Total Feature Names (unique words) TF-IDF: {len(unique_words_tfidf)}")


# Target variable (label)
y_tfidf = df['label']

train_size = 0.8  # 80% data for training
test_size = 0.2  # 20% data for testing

X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, y_tfidf, test_size=test_size, random_state=42, stratify=y_tfidf)

# Displays information about data
print("Total dataset: " + str(len(df)) + " data")
print("")
print("Training data (TF-IDF): " + str(len(X_train_tfidf)) + " (" + str(train_size * 100) + "%)")
print("Testing data (TF-IDF): " + str(len(X_test_tfidf)) + " (" + str(test_size * 100) + "%)")

# Model Random Forest(with best parameters)
rf_model = RandomForestClassifier(criterion ='gini', max_depth = 112, max_features ='log2', min_samples_split = 2, n_estimators = 125, random_state=42)

# Train the model with training data
rf_model.fit(X_train_tfidf, y_train_tfidf)

# Prediction with Random Forest model
y_pred_best_rf = rf_model.predict(X_test_tfidf)
#

print(f"Best score: {rf_model.score(X_test_tfidf, y_pred_best_rf)*100}%")

file = open('model/rf_model_AI', 'wb')

saved_rf_model_AI = pickle.dump(rf_model, file=file)

with open('model/AI_tfidf_vectorizer', 'wb') as f:
    pickle.dump(tfidf, f)


