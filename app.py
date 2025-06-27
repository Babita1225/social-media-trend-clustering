import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

st.title("Social Media Trend Clustering")

# Load CSV
df = pd.read_csv("training.csv")

# Rename column if needed
if 'clean_text' not in df.columns:
    df.columns = ['clean_text']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_text'])

# KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Evaluation
st.subheader("KMeans Evaluation")
st.write("Silhouette Score:", silhouette_score(X, labels))
st.write("Davies–Bouldin Index:", davies_bouldin_score(X.toarray(), labels))

# Plot Silhouette Score for different K
st.subheader("Choosing K using Silhouette Score")
scores = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    scores.append(silhouette_score(X, km.labels_))

fig, ax = plt.subplots()
ax.plot(range(2, 10), scores, marker='o')
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Score vs K")
st.pyplot(fig)
