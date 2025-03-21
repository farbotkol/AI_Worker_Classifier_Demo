
import streamlit as st
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample trained model (simple placeholder for demo)
def load_model():
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    job_descriptions = [
        "Full-time software engineer with benefits and superannuation",
        "Temporary contractor for IT support, paid hourly",
        "Consultant for digital transformation, milestone-based contract",
        "Freelance developer engaged under ABN",
        "12-month fixed-term marketing coordinator"
    ]
    labels = [
        "Permanent Employee",
        "Contingent Worker",
        "SOW Consultant",
        "Independent Contractor",
        "Fixed-Term Employee"
    ]
    model.fit(job_descriptions, labels)
    return model

# Load model
model = load_model()

# UI
st.title("AI Worker Classification Tool")
st.subheader("Enter a Job Description to Classify Work Type")

job_description = st.text_area("Job Description:", "Enter job details here...")

if st.button("Classify Worker"):
    if job_description.strip():
        prediction = model.predict([job_description])[0]
        prediction_prob = np.max(model.predict_proba([job_description]))
        st.success(f"Predicted Classification: {prediction}")
        st.info(f"Confidence Score: {prediction_prob:.2f}")
    else:
        st.warning("Please enter a job description.")

st.markdown("### How It Works")
st.write("This tool uses machine learning to classify workers into categories like Permanent, Contingent, SOW, Independent Contractor, or Fixed-Term.")
