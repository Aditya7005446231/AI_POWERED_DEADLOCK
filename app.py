from flask import Flask, render_template, request, jsonify
from bankers import is_safe_state, parse_input

import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/deadlock')
def deadlock_info():
    return render_template('deadlock.html')
@app.route('/predictor', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predictor.html')

    elif request.method == 'POST':
        try:
            if not request.is_json:
                return jsonify({'error': 'Request must be in JSON format'}), 400

            data = request.get_json()
            required_fields = ['allocated', 'max_need', 'available']

            if not all(field in data for field in required_fields):
                return jsonify({'error': 'Missing required fields'}), 400

            allocated = parse_input(data['allocated'], is_matrix=True)
            max_need = parse_input(data['max_need'], is_matrix=True)
            available = parse_input(data['available'], is_matrix=False)

            processes = len(allocated)
            resources = len(available)

            if len(max_need) != processes:
                return jsonify({'error': 'Allocated and Max Need matrix must have the same number of processes'}), 400

            if len(allocated[0]) != resources:
                return jsonify({'error': 'Mismatch in number of resource types'}), 400

            
            status, sequence = is_safe_state(allocated, max_need, available)

            if status == "Safe State":
                return jsonify({
                    'prediction': status,
                    'safe_sequence': sequence,
                    'status': 'success'
                })
            else:
                return jsonify({
                    'prediction': status,
                    'status': 'success'
                })

        except ValueError as e:
            return jsonify({'error': str(e), 'type': 'value_error'}), 400
        except Exception as e:
            return jsonify({'error': f'Server error: {str(e)}', 'type': 'server_error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
