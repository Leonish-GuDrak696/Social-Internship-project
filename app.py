import streamlit as st
import joblib

model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("AI-Powered Cyber Threat and Fraud Detection System")

message = st.text_area("Enter Email or Message")

if st.button("Analyze"):

    message_vec = vectorizer.transform([message])

    prediction = model.predict(message_vec)[0]
    probabilities = model.predict_proba(message_vec)[0]

    safe_prob = probabilities[0] * 100
    phishing_prob = probabilities[1] * 100

    if prediction == 1:
        st.error("⚠️ Potential Fraud / Phishing Detected")
    else:
        st.success("✅ Message Appears Safe")

    st.write(f"Safe Probability: {safe_prob:.2f}%")
    st.write(f"Phishing Probability: {phishing_prob:.2f}%")