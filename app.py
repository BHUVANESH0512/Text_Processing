import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("svm_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

label_names = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]

st.title("Emotion Detection App")

text = st.text_area("Enter your text")

if st.button("Predict"):
    
    vector = tfidf.transform([text])

    prediction = model.predict(vector)[0]

    st.success(
        f"Predicted Emotion: {label_names[prediction]}"
    )
