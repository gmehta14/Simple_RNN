import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import  sequence

#1 Load IMDB DB
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}
#reverse_word_index

# Load Model
model = load_model('simple_rnn_imdb.h5')
#model.summary()

# 2: Helper FUnctions
# function to decode reviews

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3,'?') for i in decode_review])

# function to prepare user Input
def preprocess_text(text):
    words = text.lower().split()
    encode_review = [word_index.get(word,2) + 3 for word in words]
    pedded_review = sequence.pad_sequences([encode_review],maxlen=500)
    return pedded_review

# Streamlit App

import streamlit as st

# CReate Title

st.title('IMDB Movie Review Snetiment Analysis')
st.write('Enter a movie review to clasify it as positive or negative:')

# USer Input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display Result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')

else:
    st.write('Please enter a movie review')
