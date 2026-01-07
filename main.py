from utility import load_pickle, predict_text


ai_text_predict_model = 'model/rf_model_AI'
ai_text_vectorizer = 'model/AI_tfidf_vectorizer'

fake_text_predict_model = 'model/dt_model_Fake'
fake_text_vectorizer = 'model/Fake_tfidf_vectorizer'

if __name__ == "__main__":
    ai_model = load_pickle(ai_text_predict_model)
    ai_vectorizer = load_pickle(ai_text_vectorizer)
    fake_model = load_pickle(fake_text_predict_model)
    fake_vectorizer = load_pickle(fake_text_vectorizer)
    for i in range(10):
        text = str(input("Enter the text: "))
        ai_text_predict = predict_text(text,ai_model,ai_vectorizer)[0]
        if ai_text_predict == 0:
            print("This is AI generated text")
        else:
            print("This is Human generated text")

        fake_text_predict = predict_text(text, fake_model, fake_vectorizer)[0]
        if fake_text_predict == 0:
            print("This is Fake news")
        else:
            print("This is Real news")
