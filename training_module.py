#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score,classification_report


# In[ ]:


class Training:
    def __init__(self):
        self.label_encoders = {}
    
    def encode_variables(self, data_cleaned):
        categorical_cols = data_cleaned.select_dtypes(include=['object']).columns
        
        data_encoded = data_cleaned.copy()
        
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            data_encoded[col] = self.label_encoders[col].fit_transform(data_cleaned[col])
            
        return data_encoded
    
    def split_data(self, data_encoded, target):
        print(f'Splitting {target} feature data to train models')
        X = data_encoded.drop(target, axis=1)
        y = data_encoded[target]
        return train_test_split(X, y, test_size=0.2, random_state=21)
    
    def train_and_evaluate_models(self, data_encoded, target):
        
        print(f'Training models for {target} feature')
        
        # Balancing data
        X = data_encoded.drop(target, axis=1)
        y = data_encoded[target]

        smote = SMOTE(random_state=21)
        X, y = smote.fit_resample(X, y)

        data_sm = X
        data_sm[target] = y

        # Train Test Split
        X_train, X_test, y_train, y_test = self.split_data(data_sm,target)

        # Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        
        # Initialize models
        if target == 'income':
            models = {
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Random Forest': RandomForestClassifier(),
                'Logistic Regression': LogisticRegression()
            }
        else:
            models = {
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Random Forest': RandomForestClassifier(),
                'Decision Tree': DecisionTreeClassifier(random_state=21)
            }
            
        # DataFrame to store results
        results = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'Accuracy', 'F1-Score'])
        
        # Train models and collect results
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred = pd.Series(y_pred).astype('int')
            y_test = pd.Series(y_test).astype('int')
            
            # Store results
            results = results.append({
                'Model': name,
                'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'Accuracy': accuracy_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred, average='weighted')
            }, ignore_index=True)
            
            print("\nModel:", name)
            
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            
            print("Classification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))
        
        return results

