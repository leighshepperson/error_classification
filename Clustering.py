import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import csv

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

# Read error messages from CSV file
input_file = "error_messages.csv"
output_file = "tagged_error_messages.csv"

df = pd.read_csv(input_file, header=None, names=["original_message"])
messages = df["original_message"].values.tolist()

# Preprocess error messages
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return " ".join(words)

preprocessed_messages = [preprocess_text(message) for message in messages]
df["preprocessed_message"] = preprocessed_messages

# Vectorize error messages using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_messages)

# Find the optimal number of clusters using silhouette score
best_num_clusters = 2
best_score = -1
for n_clusters in range(2, 11):  # You can adjust this range if needed
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    score = silhouette_score(X, kmeans.labels_)
    if score > best_score:
        best_num_clusters = n_clusters
        best_score = score

# Perform KMeans clustering
kmeans = KMeans(n_clusters=best_num_clusters, random_state=0).fit(X)
labels = kmeans.labels_

# Write the original error messages and their cluster labels to a new CSV file
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["original_message", "cluster"])
    for message, label in zip(messages, labels):
        writer.writerow([message, label])

print(f"Clustered error messages written to {output_file}")
