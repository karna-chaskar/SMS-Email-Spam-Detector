import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

lemma = WordNetLemmatizer()
ps = PorterStemmer()

def clean_text(text):
    tokens = word_tokenize(text.lower())

    word_tokens = [i for i in tokens if i.isalpha()]

    # stop words removal
    clean_tokens = [i for i in word_tokens if i not in stopwords.words('english')]

    # lemmatization
    lemma = WordNetLemmatizer()
    lemmatized_token = (lemma.lemmatize(i) for i in clean_tokens)

    # Stemming
    ps = PorterStemmer()
    stem_tokens = (ps.stem(i) for i in lemmatized_token)

    return " ".join(stem_tokens)

tfidf = pickle.load(open('Vectorizer.pkl','rb'))
model = pickle.load(open('Random_Forest.pkl','rb'))

st.title('Email/SMS Spam Detector')
input_sms = st.text_area('Enter the Messeage')

if st.button('Predict'):
# 1. Preprocess
    transformed_sms  = clean_text(input_sms)

# 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

# 3. Predict
    result = model.predict(vector_input)[0]

# 4. Display
    if result ==1:
        st.header('SPAM')
# 5. Display Spam Reason
        st.subheader('Why it is Spam?')
        st.caption('-Contains words similar to Spam messeges or emails .')

        wd = WordCloud(width=500, height=500, min_font_size=10, background_color='black')
        image = wd.generate(input_sms)
        st.image(image.to_image())
    else:
        st.header('NOT SPAM')