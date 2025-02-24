import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from collections import defaultdict

class DeadlockDetector:
    def __init__(self, num_processes=3, num_resources=2):
        self.num_processes = num_processes
        self.num_resources = num_resources
        self.model = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
    def parse_matrix_input(self, input_str):
        """Safely converts string matrix input to numpy array"""
        try:
            # Remove all whitespace and outer brackets
            cleaned = input_str.strip().replace('\n','').replace('\r','').replace(' ','')
            if cleaned.startswith('[[') and cleaned.endswith(']]'):
                cleaned = cleaned[2:-2]
            
            # Split into rows
            rows = cleaned.split('],[') if '],[' in cleaned else [cleaned]
            
            # Convert to 2D array
            matrix = []
            for row in rows:
                matrix.append([int(x) for x in row.split(',') if x != ''])
                
            return np.array(matrix)
        except Exception as e:
            raise ValueError(f"Invalid matrix input: {str(e)}")

    def parse_input(self, input_data):
        """Handles both string and list inputs"""
        if isinstance(input_data, str):
            try:
                # Clean string input
                cleaned = input_data.strip().replace('\n','').replace('\r','').replace(' ','')
                if cleaned.startswith('[') and cleaned.endswith(']'):
                    cleaned = cleaned[1:-1]
                return np.array([int(x) for x in cleaned.split(',') if x != ''])
            except:
                raise ValueError("Invalid input format. Use: '1,0,2,1' or [[1,0],[2,1]]")
        elif isinstance(input_data, (list, np.ndarray)):
            return np.array(input_data)
        else:
            raise ValueError("Input must be string or list")
        
            
    def generate_dataset(self, num_samples=5000):
        """Generate more realistic deadlock scenarios"""
        data = []
        
        for _ in range(num_samples):
            # Generate random system state
            allocated = np.random.randint(0, 5, (self.num_processes, self.num_resources))
            requested = np.random.randint(0, 5, (self.num_processes, self.num_resources))
            available = np.random.randint(1, 5, self.num_resources)
            
            # Calculate actual deadlock state
            is_deadlock = self.bankers_algorithm(allocated, requested, available) == "Deadlock"
            
            # Create features
            features = {
                **{f"alloc_p{i}_r{j}": allocated[i,j] for i in range(self.num_processes) 
                                      for j in range(self.num_resources)},
                **{f"request_p{i}_r{j}": requested[i,j] for i in range(self.num_processes) 
                                        for j in range(self.num_resources)},
                **{f"avail_r{j}": available[j] for j in range(self.num_resources)},
                "need_imbalance": np.sum(requested - allocated),
                "total_contention": np.sum(requested),
                "deadlock": int(is_deadlock)
            }
            data.append(features)
            
        return pd.DataFrame(data)
    
    def bankers_algorithm(self, allocated, requested, available):
        """Enhanced Banker's algorithm implementation"""
        work = np.array(available)
        need = requested - allocated
        finish = np.zeros(self.num_processes, dtype=bool)
        sequence = []
        
        for _ in range(self.num_processes):
            found = False
            for i in range(self.num_processes):
                if not finish[i] and np.all(need[i] <= work):
                    work += allocated[i]
                    finish[i] = True
                    sequence.append(i)
                    found = True
                    break
            
            if not found:
                deadlocked = np.where(~finish)[0].tolist()
                return "Deadlock", deadlocked
                
        return "Safe", sequence
    
    def train(self, num_samples=10000):
        """Train the model with enhanced features"""
        df = self.generate_dataset(num_samples)
        
        X = df.drop('deadlock', axis=1)
        y = df['deadlock']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        # Save model
        joblib.dump(self.model, 'enhanced_deadlock_model.pkl')
        print("Model trained and saved successfully")
        
    def predict(self, allocated, requested, available):
        """Make prediction with proper input handling"""
        try:
            # Convert inputs to numpy arrays
            allocated = np.array(allocated)
            requested = np.array(requested)
            available = np.array(available)
            
            # Validate shapes
            if allocated.shape != (self.num_processes, self.num_resources):
                raise ValueError(f"Allocated matrix must be {self.num_processes}x{self.num_resources}")
            if requested.shape != (self.num_processes, self.num_resources):
                raise ValueError(f"Requested matrix must be {self.num_processes}x{self.num_resources}")
            if available.shape != (self.num_resources,):
                raise ValueError(f"Available must have {self.num_resources} elements")
            
            # Banker's algorithm result
            banker_result, details = self.bankers_algorithm(allocated, requested, available)
            
            # Prepare features for ML prediction
            features = {
                **{f"alloc_p{i}_r{j}": allocated[i,j] for i in range(self.num_processes) 
                                      for j in range(self.num_resources)},
                **{f"request_p{i}_r{j}": requested[i,j] for i in range(self.num_processes) 
                                        for j in range(self.num_resources)},
                **{f"avail_r{j}": available[j] for j in range(self.num_resources)},
                "need_imbalance": np.sum(requested - allocated),
                "total_contention": np.sum(requested)
            }
            
            # Make ML prediction
            ml_pred = self.model.predict(pd.DataFrame([features]))[0]
            
            return {
                "bankers_algorithm": {
                    "status": banker_result,
                    "details": details if banker_result == "Deadlock" else {"safe_sequence": details}
                },
                "ml_prediction": "Deadlock" if ml_pred else "Safe",
                "combined_verdict": "Deadlock" if banker_result == "Deadlock" or ml_pred else "Safe",
                "confidence": self.model.predict_proba(pd.DataFrame([features]))[0][1] if ml_pred else 
                             self.model.predict_proba(pd.DataFrame([features]))[0][0]
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "recommendation": f"Please provide {self.num_processes} processes x {self.num_resources} resources matrices"
            }

def predict_deadlock(allocated, requested, available):
    """
    Standalone function to predict deadlock using the DeadlockDetector class.
    """
    detector = DeadlockDetector()
    return detector.predict(allocated, requested, available)

# Example usage
if __name__ == "__main__":
    # Initialize detector (3 processes, 2 resources by default)
    detector = DeadlockDetector()
    
    # Train the model
    print("Training model...")
    detector.train()
    
    # Example prediction
    allocated = [[1, 0], [2, 1], [0, 1]]
    requested = [[0, 1], [1, 0], [0, 0]]
    available = [2, 1]
    
    print("\nMaking prediction...")
    result = detector.predict(allocated, requested, available)
    print("\nPrediction Result:")
    import pprint
    pprint.pprint(result)