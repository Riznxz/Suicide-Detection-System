import pandas as pd  # <-- Add this line
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load preprocessed data
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
encoder = pickle.load(open('models/encoder.pkl', 'rb'))

# Load dataset again (to get labels)
df = pd.read_csv('dataset/suicide_data.csv', encoding='latin1')

# Apply text transformation
X = vectorizer.transform(df['text'].astype(str)).toarray()
y = encoder.transform(df['label'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {accuracy:.4f}")

# Save trained model
pickle.dump(model, open('models/model.pkl', 'wb'))
