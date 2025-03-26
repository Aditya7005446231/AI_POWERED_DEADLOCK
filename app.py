from flask import Flask, render_template, request
from model import predict_deadlock  # Assume model functions are in model.py

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/deadlock')
def deadlock_info():
    return render_template('deadlock.html')



@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    result = None
    safe_sequence = None
    suggestions = None
    
    if request.method == 'POST':
        allocated = request.form['allocated']
        requested = request.form['requested']
        available = request.form['available']
        
        # Assume predict_deadlock returns a tuple (result, safe_sequence, suggestions)
        result, safe_sequence, suggestions = predict_deadlock(allocated, requested, available)
    
    return render_template(
        'predictor.html',
        result=result,
        safe_sequence=safe_sequence,
        suggestions=suggestions
    )

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
