import pickle


# load pickle data
def load_pickle(location):
    file = open(location, 'rb')
    return pickle.load(file)

# predict data from given model and vectorizer
def predict_text(text, model, vectorizer):
    text_tfidf = vectorizer.transform([text])
    return model.predict(text_tfidf)
