import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re

# Download stopwords if needed
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Streamlit App Title
st.title("Tweet Clustering App 🐦🔍")
st.write("Upload a CSV of tweets and explore clustering using KMeans and DBSCAN.")

# Text Preprocessing Function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower()
    return " ".join([word for word in text.split() if word not in stop_words])

# Upload File
uploaded_file = st.file_uploader("Upload Twitter CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='latin-1', header=None)
    df.columns = ["target", "id", "date", "flag", "user", "text"]
    df['clean_text'] = df['text'].astype(str).apply(clean_text)

    st.success("File loaded and cleaned successfully!")
    sample = df.sample(5000, random_state=42)
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(sample['clean_text'])

    # Silhouette Score for KMeans
    scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        scores.append(silhouette_score(X, kmeans.labels_))

    st.subheader("Silhouette Score for KMeans Clustering")
    fig, ax = plt.subplots()
    ax.plot(range(2, 10), scores, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Optimal K for KMeans")
    st.pyplot(fig)

    # Fit KMeans with K=4 (chosen manually or based on plot)
    st.subheader("KMeans Clustering Evaluation")
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X)
    st.write("Silhouette Score (KMeans):", silhouette_score(X, labels))
    st.write("Davies–Bouldin Index (KMeans):", davies_bouldin_score(X.toarray(), labels))

    # DBSCAN Clustering
    st.subheader("DBSCAN Clustering Evaluation")
    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
    db_labels = dbscan.fit_predict(X)
    filtered = db_labels != -1
    if np.sum(filtered) > 0:
        st.write("Silhouette Score (DBSCAN):", silhouette_score(X[filtered], db_labels[filtered]))
    else:
        st.warning("DBSCAN did not find valid clusters (only noise).")