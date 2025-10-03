import streamlit as st
import pandas as pd
import joblib
from src.preprocessing import preprocess_text


model=joblib.load("models/random_forest.pkl")
tfidf_vectorizer=joblib.load("models/tfidf_vectorizer.pkl")


st.title("Fake News Detection")
st.write("Enter a news article below and the model will predict if it's Fake or Real")

user_input=st.text_area("News Article",height=200)
if st.button("predict"):
  if user_input.strip()=="":
    st.warning("Please enter some text to predict.")
  else:
    clean_input=preprocess_text(user_input)
    features=tfidf_vectorizer.transform([clean_input])
    prediction=model.predict(features)[0]
    prediction_proba=model.predict_proba(features)[0]

    if prediction==0:
      st.error("Fake News")
      st.write(f"confidence: {prediction_proba[0]*100:.2f}%")
    else:
      st.success("Real News")
      st.write(f"Confidence: {prediction_proba[1]*100:.2f}%")
  