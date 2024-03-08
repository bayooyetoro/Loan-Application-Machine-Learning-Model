import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load a Sample of the Dataset
data = pd.read_csv('./data/LoansTrainingSetV2.csv', nrows=10000)

# Optimize Data Types
for col in data.select_dtypes(include=['float64']).columns:
    data[col] = data[col].astype('float32')
for col in data.select_dtypes(include=['int64']).columns:
    data[col] = data[col].astype('int32')
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].astype('category')

# Drop irrelevant columns
data.drop(['Loan ID', 'Customer ID'], axis=1, inplace=True)

# Handle missing values
for col in data.columns:
    if data[col].dtype.name == 'category':
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].median(), inplace=True)

# Manually encode 'Loan Status'
data['Loan Status'] = data['Loan Status'].apply(lambda x: 1 if x == 'Loan Given' else 0)
data['Loan Status'] = data['Loan Status'].astype('float32')

# Encode categorical variables
categorical_cols = data.select_dtypes('category').columns.tolist()
data = pd.get_dummies(data, columns=categorical_cols)
dt = data.to_csv("data.csv",index=False)

# Split the dataset
X = data.drop('Loan Status', axis=1)
y = data['Loan Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RandomForestClassifier with Cross-Validation
rf_model = RandomForestClassifier(random_state=42)
rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
print(f'Random Forest Cross-Validation Accuracy: {np.mean(rf_cv_scores):.2%}')

# Fit the model to the entire training data after cross-validation
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f'Random Forest Test Set Accuracy: {rf_accuracy:.2%}')

# LogisticRegression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print(f'Logistic Regression Test Set Accuracy: {lr_accuracy:.2%}')
import pickle
model = pickle.open("filename.pickle", "w")