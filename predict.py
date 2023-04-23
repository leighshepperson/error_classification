import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import re


# Load the saved model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

loaded_clf = load('best_classifier_model.joblib')
vectorizer = load('vectorizer')


def preprocess_error_message(error):
    # Remove messages that start with <foo>:
    error = re.sub(r'^\w+:\s*', '', error)
    # Remove file paths (Windows and Linux)
    error = re.sub(r"([A-Za-z]:\\|\\\\|\\/)?(?:[-\w.]+(?:\\|\/))+[-\w.]+", "", error)
    # Remove numbers and special characters
    error = re.sub(r"[^a-zA-Z\s]", "", error)

    # Convert to lowercase
    error = error.lower()
    # Remove extra whitespaces
    error = " ".join(error.split())
    return error

# Load the new CSV file with error messages
new_errors_df = pd.read_csv('errors_test.csv')  # Assuming the new CSV file is named 'new_errors.csv'
new_errors_df["Category"] = new_errors_df["Category"].str.lower().str.strip()

new_errors_df["Error"] = new_errors_df["Error"].apply(preprocess_error_message)
X = vectorizer.transform(new_errors_df["Error"])


# Transform the error messages from the new CSV file
# Predict the categories using the loaded model
new_errors_df['predicted_category'] = loaded_clf.predict(X)

# Count the predicted categories and plot the counts
category_counts = new_errors_df.groupby(['predicted_category', 'Category']).size().unstack(fill_value=0)
category_counts.plot(kind='bar', stacked=True, figsize=(10, 5))

plt.xlabel('Category')
plt.ylabel('Count')
plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)

plt.title('Predicted Category Counts')
plt.show()