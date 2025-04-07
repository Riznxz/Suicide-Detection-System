import pandas as pd
import re
import nltk
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Ensure models folder exists
try:
    os.makedirs("models", exist_ok=True)
except Exception as e:
    print(f"❌ ERROR: Unable to create 'models' directory: {e}")
    exit()

# Load dataset
df = pd.read_csv('dataset/suicide_data.csv', encoding='latin1')

# Handle missing values
df['text'] = df['text'].astype(str)  
df['text'].fillna('', inplace=True)  

# Stopword set for fast lookup
stop_words = set(stopwords.words('english'))

# Data Cleaning Function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = " ".join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Apply text cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Convert text to numerical format
vectorizer = TfidfVectorizer(max_features=2000)  # Reduce to 2000 features
X = vectorizer.fit_transform(df['clean_text']).toarray().astype('float32')  # Use float32

# Check label column
label_column = 'label' if 'label' in df.columns else 'class' if 'class' in df.columns else None
if label_column is None:
    print("❌ ERROR: No valid label column found! Available columns:", df.columns)
    exit()

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(df[label_column])  # Use correct label column

# Save vectorizer and encoder
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
pickle.dump(encoder, open('models/encoder.pkl', 'wb'))

print("✅ Data preprocessing complete. Ready for model training!")
