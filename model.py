import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Function to generate simulated dataset for deadlock detection
def generate_dataset():
    # Simulating a dataset with processes, resources, allocations, requests, and deadlock outcome
    data = {
        'process_id': [1, 2, 3, 4, 5],
        'allocated_resources': [5, 3, 6, 2, 7],
        'requested_resources': [3, 2, 4, 1, 3],
        'deadlock': [0, 0, 1, 0, 1]  # 0 = No Deadlock, 1 = Deadlock
    }
    df = pd.DataFrame(data)
    return df

# Function to train the model
def train_model():
    df = generate_dataset()
    X = df[['allocated_resources', 'requested_resources']]
    y = df['deadlock']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Use Random Forest Classifier as the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, 'deadlock_model.pkl')

    return model

# Function to predict deadlock using the trained model
def predict_deadlock(allocated_resources, requested_resources):
    model = joblib.load('deadlock_model.pkl')
    
    # Predict based on input resources
    prediction = model.predict([[allocated_resources, requested_resources]])
    return prediction[0]
