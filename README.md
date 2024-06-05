import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df1 = pd.read_csv("C:\\Users\\LENOVO\\Desktop\\project\\spam\\spam.csv",encoding='iso-8859-1')

# Assuming the dataset has columns 'label' and 'text'
print(df1.head())
df1.shape
df1.info()
df1.drop(columns=['Unnamed: 2', 'Unnamed: 3','Unnamed: 4'],inplace=True)

# Rename the columns to match the expected names for the classifier
df1.rename(columns = {'v1':'label', 'v2':'text'},inplace=True)
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert text to lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    return text

# Apply the preprocessing function to the emails
df1['text'] = df1['text'].apply(preprocess_text)

import nltk
!pip install nltk
nltk.download('punkt')
df1['num_characters']=df1['text'].apply(len)

df1['num_words']=df1['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df1.head()

# Convert labels to binary values
df1['label'] = df1['label'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df1['text'], df1['label'], test_size=0.2, random_state=42)

# Transform the text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the model
model = MultinomialNB()

# Train the model
model.fit(X_train_tfidf, y_train)

# Predict the labels for the test set
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Display the classification report
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# Function to classify new emails
def classify_email(email):
    email = preprocess_text(email)
    email_tfidf = vectorizer.transform([email])
    prediction = model.predict(email_tfidf)
    return 'spam' if prediction[0] == 1 else 'ham'

# Example emails to classify
emails = ["Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/123456 to claim now.", "Hi team, Please find the project report attached. Let me know if you have any questions."]
for email in emails:
    print(f'Email: "{email}" is classified as: {classify_email(email)}')

