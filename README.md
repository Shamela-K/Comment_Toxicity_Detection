# ⚠️ Deep Learning Comment Toxicity Detection

## 📌 Project Overview

This project detects toxic comments using Natural Language Processing (NLP) and Deep Learning techniques.

The model classifies comments into multiple toxicity categories such as:

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

The project also includes an interactive Streamlit web application for real-time predictions and bulk CSV predictions.

---

## 🚀 Technologies Used

- Python
- TensorFlow
- Keras
- NLP
- NLTK
- Pandas
- NumPy
- Streamlit
- Matplotlib
- Seaborn

---

## 📊 Dataset

Dataset used:
Jigsaw Toxic Comment Classification Dataset

Dataset contains:
- 159,571 comments
- Multi-label toxicity classification

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Lowercasing
- Removing URLs
- Removing punctuation
- Removing stopwords
- Text cleaning using Regular Expressions
- Tokenization
- Text Vectorization

---

## 🧠 Deep Learning Model

Model Architecture:
- Embedding Layer
- Bidirectional LSTM
- Dense Layers
- Sigmoid Output Layer

Loss Function:
- Binary Crossentropy

Optimizer:
- Adam

---

## 📈 Model Performance

- Training Accuracy: 99.33%
- Validation Accuracy: 99.30%

---

## 💻 Streamlit Application Features

✅ Real-time toxicity prediction  
✅ Bulk CSV prediction  
✅ Download prediction results  
✅ Interactive dashboard  
✅ Sample test comments  

---

## ▶️ How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Shamela-K/Comment_Toxicity_Detection.git

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App

```bash
streamlit run app.py
```

## 🎯 Sample Prediction

Input:
```text
I hate you idiot
```

Output:
```text
toxic : detected
insult : detected
```

## 📂 Project Structure

```bash
Comment_Toxicity_Detection/
│
├── app.py
├── comment_toxicity_dl.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## 👨‍💻 Developed By

Shamela K