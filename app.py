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
    if request.method == 'POST':
        allocated_resources = request.form['allocated_resources']
        requested_resources = request.form['requested_resources']
        prediction = predict_deadlock(allocated_resources, requested_resources)
        return render_template('predictor.html', prediction=prediction)
    return render_template('predictor.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
