from flask import Flask, render_template, request
from model import BankersAlgorithm

app = Flask(__name__)
banker = BankersAlgorithm()

def convert_to_positive(value):
    """Convert any number to its absolute value (positive)"""
    try:
        return abs(int(value))
    except (ValueError, TypeError):
        return 0  # Default to 0 if conversion fails

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    if request.method == 'POST':
        try:
            # Get and convert to positive values
            n_processes = convert_to_positive(request.form.get('n_processes', 5))
            n_resources = convert_to_positive(request.form.get('n_resources', 3))
            
            # Ensure at least 1 process and resource
            n_processes = max(1, n_processes)
            n_resources = max(1, n_resources)
            
            # Parse matrices with automatic conversion to positive
            allocation = []
            max_demand = []
            for i in range(n_processes):
                alloc_row = []
                max_row = []
                for j in range(n_resources):
                    alloc_val = convert_to_positive(request.form.get(f'allocation_{i}_{j}', 0))
                    max_val = convert_to_positive(request.form.get(f'max_{i}_{j}', 0))
                    alloc_row.append(alloc_val)
                    max_row.append(max_val)
                allocation.append(alloc_row)
                max_demand.append(max_row)
            
            # Parse available resources with conversion to positive
            available = []
            for j in range(n_resources):
                avail_val = convert_to_positive(request.form.get(f'avail_{j}', 0))
                available.append(avail_val)
            
            # Run Banker's Algorithm
            banker.set_data(allocation, max_demand, available)
            is_safe, sequence = banker.check_safety()
            
            return render_template('predictor.html',
                                safe=is_safe,
                                sequence=banker.format_sequence(sequence),
                                allocation=allocation,
                                max_demand=max_demand,
                                available=available,
                                n_processes=n_processes,
                                n_resources=n_resources)
        
        except Exception as e:
            return render_template('predictor.html', 
                                error=f"An error occurred: {str(e)}",
                                n_processes=request.form.get('n_processes', 5),
                                n_resources=request.form.get('n_resources', 3))
    
    # GET request - provide defaults
    return render_template('predictor.html',
                         n_processes=5,
                         n_resources=3)

@app.route('/deadlock')
def deadlock():
    return render_template('deadlock.html')

if __name__ == '__main__':
    app.run(debug=True)