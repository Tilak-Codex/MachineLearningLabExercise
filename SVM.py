import numpy as np import pandas as pd
import matplotlib.pyplot as plt import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv(r"..\data\spamhamdata.csv", sep="\t", header=None, names=["label", "text"]) df = df.dropna()

print("Shape of dataset:", df.shape) print("Target classes:", df["label"].unique())

df.head() sns.countplot(x=df["label"]) plt.title("Class Distribution") plt.show()
df["length"] = df["text"].str.len()

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="length", hue="label", bins=60, element="step") plt.title("Message Length Distribution")
plt.show()
X_train, X_test, y_train, y_test = train_test_split(
df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
 
vectorizer = TfidfVectorizer(stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train) X_test_vec = vectorizer.transform(X_test) svm_model = LinearSVC() svm_model.fit(X_train_vec, y_train)
y_pred = svm_model.predict(X_test_vec) accuracy = accuracy_score(y_test, y_pred) print("Accuracy:", accuracy)
labels = ["ham", "spam"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels) plt.title("Confusion Matrix")
plt.xlabel("Predicted") plt.ylabel("Actual") plt.show()
print(classification_report(y_test, y_pred)) sample_texts = [
"Congratulations! You have won a free ticket. Reply WIN now.", "Hey, are we still meeting for lunch today?",
]

sample_vec = vectorizer.transform(sample_texts) sample_pred = svm_model.predict(sample_vec)

for text, label in zip(sample_texts, sample_pred): print(f"{label.upper()}: {text}")

 
app.py:
 


import numpy as np import pandas as pd import streamlit as st from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer from sklearn.svm import LinearSVC

st.set_page_config(page_title="SVM Spam Detector", page_icon=":email:", layout="wide") @st.cache_resource
def build_model():
data_path = Path(  file  ).parent.parent / "data" / "spamhamdata.csv"

df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "text"]) df = df.dropna()

X_train, _, y_train, _ = train_test_split(
df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
 
)
vectorizer = TfidfVectorizer(stop_words="english") X_train_vec = vectorizer.fit_transform(X_train)
 
model = LinearSVC() model.fit(X_train_vec, y_train)

return df, vectorizer, model

df, vectorizer, model = build_model()

st.markdown("<div class='title'>SVM Spam Detector</div>", unsafe_allow_html=True) st.markdown(
"<div class='subtle'>Detect spam emails with a support vector machine trained on the spam/ham dataset.</div>",
unsafe_allow_html=True,
)

st.markdown("<span class='pill'>Model: Linear SVM</span><span class='pill'>TF-IDF</span>", unsafe_allow_html=True)
col_left, col_right = st.columns([1.2, 0.8], gap="large") with col_left:
st.markdown("<div class='card'>", unsafe_allow_html=True) st.subheader("Email Content")
st.caption("Paste a message to classify as spam or ham.")

default_text = "Congratulations! You have won a free ticket. Reply WIN now." message = st.text_area("Message text", value=default_text, height=180)
st.markdown("</div>", unsafe_allow_html=True) with col_right:
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Prediction")
st.caption("The model returns the class and a confidence score.")

if message.strip():
vector = vectorizer.transform([message]) pred = model.predict(vector)[0]
score = float(model.decision_function(vector)[0]) confidence = 1.0 / (1.0 + np.exp(-abs(score)))
label = "Spam" if pred == "spam" else "Ham"
color = "#dc3545" if pred == "spam" else "#28a745" # Red for spam, green for ham

st.markdown(f"""
<div style="margin: 20px 0;">
<div style="color: #6c757d; font-size: 0.875rem; margin-bottom: 4px;">Prediction</div>
<div style="color: {color}; font-size: 2.5rem; font-weight: 600; margin-bottom: 10px;">{label}</div>
</div>
""", unsafe_allow_html=True)
st.progress(confidence, text=f"Confidence: {confidence:.2%}") st.write("Decision score:", score)
else:
st.info("Enter a message to get a prediction.")
 
User Interface:
 
st.markdown("</div>", unsafe_allow_html=True) st.markdown("---")
with st.expander("Dataset overview"): st.write("Rows:", len(df))
st.write("Labels:", sorted(df["label"].unique().tolist())) st.dataframe(df.head())
st.markdown("<div class='card'>", unsafe_allow_html=True) st.markdown("</div>", unsafe_allow_html=True)