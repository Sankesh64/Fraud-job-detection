import streamlit as st
import joblib
import re
import string
import spacy
import numpy as np
import scipy.sparse as sp
import subprocess
import sys

# ---------------------------------------------
# Load model
# ---------------------------------------------

model = joblib.load("fraud_detector_xgboost.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
encoders = joblib.load("label_encoders.pkl")

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

# ---------------------------------------------
# Text Cleaning Function
# ---------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    doc = nlp(text)
    words = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_space
        and not token.is_punct
    ]
    return " ".join(words)

# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------

st.set_page_config(page_title="Fake Job Detector", page_icon="💼")

st.title("💼 Fake Job Detection System")

st.write("Paste a job description and provide job details to predict whether it is Fraudulent or Legitimate.")

col1, col2 = st.columns(2)

with col1:
    telecommuting = st.checkbox("Telecommuting")
    has_company_logo = st.checkbox("Has Company Logo", value=True)
    has_questions = st.checkbox("Has Screening Questions")
    
    employment_type = st.selectbox("Employment Type", encoders['employment_type'].classes_, index=list(encoders['employment_type'].classes_).index('Unknown') if 'Unknown' in encoders['employment_type'].classes_ else 0)
    required_experience = st.selectbox("Required Experience", encoders['required_experience'].classes_, index=list(encoders['required_experience'].classes_).index('Unknown') if 'Unknown' in encoders['required_experience'].classes_ else 0)

with col2:
    required_education = st.selectbox("Required Education", encoders['required_education'].classes_, index=list(encoders['required_education'].classes_).index('Unknown') if 'Unknown' in encoders['required_education'].classes_ else 0)
    industry = st.selectbox("Industry", encoders['industry'].classes_, index=list(encoders['industry'].classes_).index('Unknown') if 'Unknown' in encoders['industry'].classes_ else 0)
    function = st.selectbox("Function", encoders['function'].classes_, index=list(encoders['function'].classes_).index('Unknown') if 'Unknown' in encoders['function'].classes_ else 0)

job_text = st.text_area("Enter Job Description", height=200)

if st.button("Predict"):
    if job_text.strip() == "":
        st.warning("Please enter a job description.")
    else:
        cleaned = clean_text(job_text)
        text_vector = tfidf.transform([cleaned])
        
        # Encode categorical features
        emp_type_enc = encoders['employment_type'].transform([employment_type])[0]
        req_exp_enc = encoders['required_experience'].transform([required_experience])[0]
        req_edu_enc = encoders['required_education'].transform([required_education])[0]
        ind_enc = encoders['industry'].transform([industry])[0]
        func_enc = encoders['function'].transform([function])[0]
        
        # Combine additional features in the expected order:
        # [telecommuting, has_company_logo, has_questions, employment_type, required_experience, required_education, industry, function]
        additional_features = np.array([[
            int(telecommuting),
            int(has_company_logo),
            int(has_questions),
            emp_type_enc,
            req_exp_enc,
            req_edu_enc,
            ind_enc,
            func_enc
        ]])
        
        # Stack TF-IDF features with additional features
        final_vector = sp.hstack([text_vector, additional_features])

        prediction = model.predict(final_vector)[0]
        probability = model.predict_proba(final_vector)[0]
        confidence = probability[prediction] * 100

        if prediction == 1:
            st.error("Fraudulent Job")
        else:
            st.success("Legitimate Job")

        st.write(f"Confidence : **{confidence:.2f}%**")
        st.progress(float(confidence / 100))
