import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Load dataset
df = pd.read_csv('Dataset of Diabetes .csv')

# Streamlit title and description
st.title("Diabetes Prediction Model")
st.write("This app predicts diabetes based on various health features.")

# Display first 5 rows of the dataset
st.subheader("First 5 Rows of Dataset")
st.write(df.head())

# Check for null values and display
null_count = df.isnull().sum()
st.subheader("Count of Null Values")
st.write(null_count)

# Clean the data
df['CLASS'] = df['CLASS'].str.strip()
df['Gender'] = df['Gender'].str.upper()
df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})

# One-hot encoding for CLASS
df = pd.get_dummies(df, columns=['CLASS'], drop_first=True)

# Drop unnecessary columns
df = df.drop(['ID', 'No_Pation'], axis=1)

# Create the target column
df['Target'] = df[['CLASS_P', 'CLASS_Y']].idxmax(axis=1)
df = df.drop(['CLASS_P', 'CLASS_Y'], axis=1)

# Label encode the target
encoder = LabelEncoder()
df['Target'] = encoder.fit_transform(df['Target'])

# Features and labels
X = df.drop(['Target'], axis=1)
y = df['Target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Accuracy results
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

st.subheader(f"Training Accuracy: {train_accuracy:.2f}")
st.subheader(f"Testing Accuracy: {test_accuracy:.2f}")

# Classification report
st.subheader("Classification Report (Test Data)")
st.text(classification_report(y_test, y_test_pred))

# Feature importances
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

st.subheader("Feature Importance")
st.write(importance_df)

# Feature importance plot
st.subheader("Feature Importance Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df['Feature'], importance_df['Importance'])
ax.set_xlabel('Importance')
ax.set_ylabel('Features')
ax.set_title('Feature Importances')
st.pyplot(fig)

# User input for prediction
st.subheader("Make a Prediction")
input_data = []
for col in X.columns:
    input_value = st.number_input(f"Enter {col}", value=0.0)
    input_data.append(input_value)

if st.button("Predict"):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    predicted_class = encoder.inverse_transform(prediction)
    st.write(f"Predicted Class: {predicted_class[0]}")
