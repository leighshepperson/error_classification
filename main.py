import re
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score

# Function to preprocess the error message
from sklearn.tree import DecisionTreeClassifier


def preprocess_error_message(error):
    # Remove file paths (Windows and Linux)
    error = re.sub(r"([A-Za-z]:\\|\\\\|\\/)?(?:[-\w.]+(?:\\|\/))+[-\w.]+", "", error)
    # Remove numbers and special characters
    error = re.sub(r"[^a-zA-Z\s]", "", error)
    # Convert to lowercase
    error = error.lower()
    # Remove extra whitespaces
    error = " ".join(error.split())
    return error


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix {model_name}")
    plt.show()


def plot_error_category_counts(y_pred, model_name):
    category_counts = y_pred.value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title(f"{model_name} Predicted Error Category Counts")
    plt.show()


# Read the errors from the CSV file
data = pd.read_csv("errors.csv")
data["Category"] = data["Category"].str.lower().str.strip()

category_counts = data["Category"].value_counts()

# Convert the category_counts Series to a DataFrame
category_counts_df = pd.DataFrame(category_counts).reset_index()
category_counts_df.columns = ["Category", "Count"]

print(category_counts_df)
# Extract error messages and their respective categories
error_messages = data["Error"].apply(preprocess_error_message)
categories = data["Category"]

# Vectorize the error messages
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(error_messages)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, categories, test_size=0.2, random_state=42
)

param_distributions = {
    "Logistic Regression": {
        "C": np.logspace(-4, 4, 20),
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    },
    "Linear SVM": {"C": np.logspace(-4, 4, 20)},
    "Random Forest": {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 6],
    },
    "Multinomial Naive Bayes": {
        "alpha": np.linspace(0.1, 1, 10),
    },
    "K-Nearest Neighbors": {
        "n_neighbors": list(range(1, 21)),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    },
    "Decision Trees": {
        "criterion": ["gini", "entropy"],
        "max_depth": [None] + list(range(1, 21)),
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "SVM with RBF Kernel": {
        "C": np.logspace(-4, 4, 20),
        "gamma": ["scale", "auto"] + list(np.logspace(-4, 4, 20)),
    },
}

# Train and evaluate classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(solver="saga", max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(),
    "Multinomial Naive Bayes": MultinomialNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Trees": DecisionTreeClassifier(),
    "SVM with RBF Kernel": SVC(kernel="rbf"),
}

param_grids = {
    "Logistic Regression": {
        "C": np.logspace(-4, 4, 20),
    },
    "Linear SVM": {
        "C": np.logspace(-4, 4, 20),
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "Multinomial Naive Bayes": {
        "alpha": np.linspace(0.1, 1, 10),
    },
    "K-Nearest Neighbors": {
        "n_neighbors": list(range(1, 21)),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    },
    "Decision Trees": {
        "criterion": ["gini", "entropy"],
        "max_depth": [None] + list(range(1, 21)),
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "SVM with RBF Kernel": {
        "C": np.logspace(-4, 4, 20),
        "gamma": ["scale", "auto"] + list(np.logspace(-4, 4, 20)),
    },
}


for name, clf in classifiers.items():
    # Initialize GridSearchCV with the classifier, parameter grid, and scoring metric
    grid_search = GridSearchCV(
        clf, param_grids[name], scoring="f1_weighted", cv=5, verbose=1, n_jobs=-1
    )

    # Fit GridSearchCV with the training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best F1-score
    best_params = grid_search.best_params_
    best_f1_score = grid_search.best_score_

    print(f"{name} - Best parameters: {best_params}")
    print(f"{name} - Best F1-score: {best_f1_score}")

    # Evaluate the best model on the test data
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print(f"{name} - Classification Report:")
    print(classification_report(y_test, y_pred))


for name, clf in classifiers.items():
    random_search = RandomizedSearchCV(
        clf,
        param_distributions=param_distributions[name],
        scoring="f1_weighted",
        cv=5,
        n_iter=20,  # Number of parameter settings sampled
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    best_f1_score = random_search.best_score_

    print(f"{name} Best parameters: {best_params}")
    print(f"{name} Best F1-score: {best_f1_score}\n")

    # Evaluate the best model on the test data
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))

best_classifier = None
best_f1_score = 0
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, labels=categories.unique(), model_name=name)
    plot_error_category_counts(pd.Series(y_pred), model_name=name)

    current_f1_score = f1_score(y_test, y_pred, average="weighted")
    print(f"{name} Weighted F1-score: {current_f1_score}\n")

    if current_f1_score > best_f1_score:
        best_f1_score = current_f1_score
        best_classifier = name

print(best_classifier, best_f1_score)
