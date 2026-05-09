# =========================================================
# Deep Learning Comment Toxicity Detection - Streamlit App
# =========================================================

# -----------------------------
# Import Libraries
# -----------------------------

import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import re

from nltk.corpus import stopwords
from tensorflow.keras.layers import TextVectorization

# -----------------------------
# Streamlit Page Config
# -----------------------------

st.set_page_config(
    page_title="Toxicity Detection",
    page_icon="⚠️",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------

model = tf.keras.models.load_model(
    "toxicity_model.h5"
)

# -----------------------------
# Load Vectorizer
# -----------------------------

with open("vectorizer.pkl", "rb") as f:
    vectorizer_data = pickle.load(f)

vectorizer = TextVectorization.from_config(
    vectorizer_data["config"]
)

vectorizer.set_weights(
    vectorizer_data["weights"]
)

# -----------------------------
# Stopwords
# -----------------------------

stop_words = set(stopwords.words('english'))

# -----------------------------
# Text Cleaning Function
# -----------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -----------------------------
# Toxicity Labels
# -----------------------------

labels = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate'
]

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 Project Information")

st.sidebar.markdown("""
### Technologies Used
- Deep Learning
- NLP
- TensorFlow
- Streamlit

### Model
Bidirectional LSTM

### Dataset
Jigsaw Toxic Comment Dataset
""")

# =========================================================
# TITLE
# =========================================================

st.markdown("""
# ⚠️ Deep Learning Comment Toxicity Detection

Detect toxic online comments using NLP and Deep Learning.
""")

# =========================================================
# DATASET INSIGHTS
# =========================================================

st.header("📊 Dataset Insights")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Dataset Size",
    "159,571"
)

col2.metric(
    "Categories",
    "6"
)

col3.metric(
    "Validation Accuracy",
    "99.3%"
)

# =========================================================
# TOXICITY CATEGORIES
# =========================================================

st.subheader("Toxicity Categories")

st.markdown("""
- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate
""")

# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.header("📈 Model Performance")

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Training Accuracy",
    "99.33%"
)

metric2.metric(
    "Validation Accuracy",
    "99.30%"
)

metric3.metric(
    "Validation Loss",
    "0.0376"
)

# =========================================================
# SAMPLE TEST COMMENTS
# =========================================================

st.header("💡 Sample Test Comments")

st.info("I hate you idiot")

st.success("Have a nice day")

st.warning("You are a stupid person")

# =========================================================
# SINGLE COMMENT PREDICTION
# =========================================================

st.header("🧠 Single Comment Prediction")

user_input = st.text_area(
    "Enter a Comment"
)

if st.button("Predict Toxicity"):

    if user_input.strip() != "":

        # Clean text

        cleaned_text = clean_text(user_input)

        # Vectorize

        vectorized_text = vectorizer(
            [cleaned_text]
        )

        # Predict

        prediction = model.predict(
            vectorized_text
        )

        st.subheader("Prediction Results")

        # Show probabilities

        for label, prob in zip(labels, prediction[0]):

            percentage = round(prob * 100, 2)

            st.write(f"### {label}")

            st.progress(float(prob))

            st.write(f"{percentage}%")

    else:

        st.warning(
            "Please enter a comment"
        )

# =========================================================
# BULK CSV PREDICTION
# =========================================================

st.header("📂 Bulk Prediction using CSV")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    bulk_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(
        bulk_df.head(),
        use_container_width=True
    )

    # Check required column

    if 'comment_text' in bulk_df.columns:

        # Clean comments

        bulk_df['clean_comment'] = (
            bulk_df['comment_text']
            .apply(clean_text)
        )

        # Vectorize

        bulk_vector = vectorizer(
            bulk_df['clean_comment'].values
        )

        # Predict

        bulk_predictions = model.predict(
            bulk_vector
        )

        # Convert to binary

        binary_predictions = (
            bulk_predictions > 0.5
        ).astype(int)

        prediction_df = pd.DataFrame(
            binary_predictions,
            columns=labels
        )

        # Final output

        final_output = pd.concat(
            [
                bulk_df,
                prediction_df
            ],
            axis=1
        )

        st.subheader("Prediction Preview")

        st.dataframe(
            final_output.head(10),
            use_container_width=True
        )

        # Download predictions

        csv = final_output.to_csv(
            index=False
        ).encode('utf-8')

        st.download_button(
            label="⬇ Download Predictions CSV",
            data=csv,
            file_name='bulk_predictions.csv',
            mime='text/csv'
        )

    else:

        st.error(
            "CSV file must contain 'comment_text' column"
        )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    "Developed using Deep Learning, NLP, TensorFlow, and Streamlit 🚀"
)