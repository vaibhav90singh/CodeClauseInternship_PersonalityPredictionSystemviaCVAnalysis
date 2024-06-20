import spacy
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK resources
nltk.download('punkt')

# Load dataset (replace with your actual dataset)
# Example dataset structure: df should have columns like 'resume_text', 'extroversion', 'conscientiousness', etc.
df = pd.read_csv('cv_dataset.csv')

# Assuming 'extroversion' is one of the personality traits we want to predict
X = df['resume_text']
y = df['extroversion']  # Replace 'extroversion' with the personality trait you want to predict

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the NLP pipeline using CountVectorizer and Logistic Regression
pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(max_iter=1000)),
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
