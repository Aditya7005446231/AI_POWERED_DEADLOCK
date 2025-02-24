from flask import Flask, render_template, request
from model import DeadlockDetector

app = Flask(__name__)

detector = DeadlockDetector()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/deadlock')
def deadlock_info():
    return render_template('deadlock.html')



@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    if request.method == 'POST':
        try:
            allocated = [list(map(int, row.split(','))) for row in request.form['allocated'].split(';')]
            requested = [list(map(int, row.split(','))) for row in request.form['requested'].split(';')]
            available = list(map(int, request.form['available'].split(',')))

            result = detector.predict(allocated, requested, available)

            return render_template('predictor.html', result=result["combined_verdict"],
                                   safe_sequence=result["bankers_algorithm"]["details"].get("safe_sequence", []),
                                   suggestions=result.get("recommendations", []))
        except Exception as e:
            return render_template('predictor.html', result=f"Error: {str(e)}", safe_sequence=[], suggestions=[])
    else:
        # Handle GET request by rendering an empty form
        return render_template('predictor.html', result=None, safe_sequence=[], suggestions=[])


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
