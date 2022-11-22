import streamlit as st
import pickle
import numpy as np
from tensorflow import keras
from keras.utils.data_utils import pad_sequences

lstm_model = keras.models.load_model('nextword_model.h5')
t = pickle.load(open('nextword_tokenizer.pkl','rb'))

def predict_word(text,model,tokenizer):
    encoded_text = t.texts_to_sequences([text])[0]       

    encoded_text = pad_sequences([encoded_text], maxlen=5, padding='post')

    yhat = np.argmax(lstm_model.predict(encoded_text), axis=-1) 

    out_word = ''

    for word, index in tokenizer.word_index.items():
        if index == yhat:
            out_word = word
            break

    return out_word


st.title("Prediction the next word in a sequence")
st.markdown("Here we are predicting the next word from an input of 5 words")

st.subheader("Enter 5 words separated by a space")
seed_text = st.text_input('input')

if st.button('Predict Next Word'):
    pred_word = predict_word(seed_text,lstm_model,t)
    st.write('Predicted word:', pred_word)
    




