import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


## Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

##load model h5 file
model = load_model('simple_rnn_imdb.h5')

## step 2 Helpter function to preprocess input text
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    enocded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([enocded_review], maxlen=500)
    return padded_review

## streamlit app

import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment (Positive/Negative).")

user_input = st.text_area("Movie Review", height=200)

if st.button("Classify"):
    preprocess_input = preprocess_text(user_input)

    ## make prediction
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
    score = prediction[0][0]
    st.write(f"Predicted Sentiment: **{sentiment}**")
    st.write(f"Prediction Score: **{score:.4f}**")  
else:
    st.write("Please enter a movie review and click 'Classify' to see the sentiment prediction.")
