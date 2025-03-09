from flask import Flask, render_template, request
from model import train_model, predict_deadlock

app = Flask(__name__)

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle model training
@app.route('/train', methods=['POST'])
def train():
    train_model()
    return "Model trained successfully!"

# Route to handle deadlock prediction
@app.route('/predict', methods=['POST'])
def predict():
    allocated_resources = int(request.form['allocated_resources'])
    requested_resources = int(request.form['requested_resources'])
    
    # Call prediction function
    result = predict_deadlock(allocated_resources, requested_resources)
    
    if result == 1:
        return "Deadlock detected!"
    else:
        return "No deadlock detected."

if __name__ == '__main__':
    app.run(debug=True)
