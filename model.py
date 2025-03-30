import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def generate_realistic_dataset(num_samples=1000):
    """Generate a more realistic deadlock dataset"""
    np.random.seed(42)
    
    data = {
        'allocated_R1': np.random.randint(0, 10, num_samples),
        'allocated_R2': np.random.randint(0, 10, num_samples),
        'requested_R1': np.random.randint(0, 10, num_samples),
        'requested_R2': np.random.randint(0, 10, num_samples),
        'available_R1': np.random.randint(1, 15, num_samples),
        'available_R2': np.random.randint(1, 15, num_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Simulate deadlock conditions (Banker's algorithm logic)
    df['deadlock'] = np.where(
        (df['requested_R1'] > df['available_R1']) | 
        (df['requested_R2'] > df['available_R2']) |
        ((df['allocated_R1'] + df['allocated_R2']) > (df['available_R1'] + df['available_R2'])),
        1, 0
    )
    
    return df

def train_and_evaluate_model():
    df = generate_realistic_dataset(5000)
    
    # Features: allocated, requested, and available resources
    X = df[['allocated_R1', 'allocated_R2', 'requested_R1', 'requested_R2', 'available_R1', 'available_R2']]
    y = df['deadlock']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    joblib.dump(model, 'deadlock_model.pkl')
    return model

def predict_deadlock(allocated, requested, available):
    """Predict deadlock using trained model"""
    try:
        # Load trained model
        model = joblib.load('deadlock_model.pkl')
        
        # Prepare input data (convert from strings to arrays if needed)
        if isinstance(allocated, str):
            allocated = [int(x) for x in allocated.strip('[]').split(',')]
        if isinstance(requested, str):
            requested = [int(x) for x in requested.strip('[]').split(',')]
        if isinstance(available, str):
            available = [int(x) for x in available.strip('[]').split(',')]
        
        # Create feature vector (assuming 2 resource types)
        features = [
            allocated[0], allocated[1],  # allocated R1, R2
            requested[0], requested[1],  # requested R1, R2
            available[0], available[1]   # available R1, R2
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Generate results
        if prediction == 0:
            result = "System is in a safe state"
            safe_seq = find_safe_sequence(allocated, requested, available)
            recommendations = ["All resources can be safely allocated"]
        else:
            result = "System is in an unsafe state (potential deadlock)"
            safe_seq = []
            recommendations = [
                "Reduce resource requests",
                "Increase available resources",
                "Terminate some processes"
            ]
        
        return result, safe_seq, recommendations
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return "Error in prediction", [], ["Check your input format"]

def find_safe_sequence(allocated, requested, available):
    """Simplified safe sequence finder (Banker's algorithm)"""
    # This is a placeholder - implement proper Banker's algorithm here
    return [0, 1, 2] if sum(available) > sum(requested) else []