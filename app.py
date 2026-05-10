
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Comment Toxicity Detection",
    page_icon="⚠️",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }

    .title {
        text-align: center;
        font-size: 45px;
        font-weight: bold;
        color: #ff4b4b;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #444444;
        margin-bottom: 30px;
    }

    .result-box {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
        margin-top: 20px;
    }

    .footer {
        text-align: center;
        color: gray;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = load_model("toxicity_model.h5")

# ---------------------------------------------------
# LOAD VECTORIZER
# ---------------------------------------------------

with open("vectorizer.pkl", "rb") as f:
    vectorizer_data = pickle.load(f)

vectorizer = TextVectorization.from_config(
    vectorizer_data["config"]
)

vectorizer.adapt(["test"])

vectorizer.set_weights(
    vectorizer_data["weights"]
)

# ---------------------------------------------------
# LABELS
# ---------------------------------------------------

labels = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate'
]

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    '<div class="title">⚠️ Comment Toxicity Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deep Learning + NLP based Toxic Comment Classifier</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# LAYOUT
# ---------------------------------------------------

col1, col2 = st.columns([2, 1])

# ---------------------------------------------------
# LEFT SIDE
# ---------------------------------------------------

with col1:

    st.subheader("✍️ Enter Comment")

    user_input = st.text_area(
        "",
        placeholder="Type your comment here...",
        height=180
    )

    predict_button = st.button("🚀 Predict Toxicity")

# ---------------------------------------------------
# RIGHT SIDE
# ---------------------------------------------------

with col2:

    st.subheader("📌 Sample Comments")

    st.info("Have a nice day 😊")
    st.warning("I hate you idiot")
    st.success("This project is amazing")

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if predict_button:

    if user_input.strip() == "":
        st.warning("Please enter a comment.")

    else:

        vectorized_text = vectorizer([user_input])

        prediction = model.predict(vectorized_text)

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.subheader("📊 Prediction Results")

        for i, label in enumerate(labels):

            score = float(prediction[0][i] * 100)

            st.write(f"### {label.replace('_',' ').title()} : {score:.2f}%")

            st.progress(min(int(score), 100))
            if score < 10:
               st.success("Not Detected")
            else:
               st.error(f"{score:.2f}% Detected")

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# CSV UPLOAD
# ---------------------------------------------------

st.markdown("---")

st.subheader("📂 Bulk CSV Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("### Uploaded Dataset")
    st.dataframe(df.head())

    if "comment_text" in df.columns:

        vectorized_comments = vectorizer(
            df["comment_text"].values
        )

        predictions = model.predict(vectorized_comments)

        predictions_df = pd.DataFrame(
            predictions,
            columns=labels
        )

        final_df = pd.concat(
            [df, predictions_df],
            axis=1
        )

        st.write("### Prediction Results")
        st.dataframe(final_df.head())

        csv = final_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Predictions",
            data=csv,
            file_name="test_predictions.csv",
            mime="text/csv"
        )

    else:
        st.error("CSV must contain 'comment_text' column.")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("📘 About Project")

st.sidebar.info(
    """
    This project uses:

    ✅ NLP
    ✅ Deep Learning
    ✅ Bidirectional LSTM
    ✅ TensorFlow
    ✅ Streamlit

    to detect toxic comments.
    """
)

st.sidebar.success("Model Accuracy: 99.3%")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown(
    '<div class="footer">Developed by Shamela K 🚀</div>',
    unsafe_allow_html=True
)
