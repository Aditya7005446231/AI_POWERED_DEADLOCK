# 🔄 AI-Powered Deadlock Detection System

**Smarter deadlock prediction combining Banker's Algorithm with Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)

## 🌟 Features

- **Hybrid Detection Engine**
  - Traditional Banker's Algorithm for guaranteed correctness
  - Machine Learning model (Logistic Regression/SVM) for probability estimation
  - Explainable AI insights into deadlock conditions

- **Simple Interface**
  - Easy matrix input for resource allocation
  - Instant "Deadlock/No Deadlock" results
  - Clean visualization of resource states

- **Educational Value**
  - Built-in deadlock explanations
  - Example scenarios with walkthroughs
  - Beginner-friendly documentation

## 🛠️ Tech Stack

| Component       | Technology |
|----------------|------------|
| Backend        | Python, Flask |
| Frontend       | HTML5, CSS3, Jinja2 |
| Machine Learning | scikit-learn, Pandas |
| Visualization  | Matplotlib (basic graphs) |

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/AI_POWERED_DEADLOCK.git
   cd AI_POWERED_DEADLOCK

   
2. **Install dependencies**
   pip install -r requirements.txt

3. Run the application
   python app.py

4. Access the web interface
   http://127.0.0.1:5000


**📂 Project Structure**

AI_POWERED_DEADLOCK/
├── app.py                # Flask application
├── model.py              # ML model and Banker's Algorithm
├── templates/            # UI templates
│   ├── index.html        # Main interface
│   ├── results.html      # Analysis results
│   └── explain.html      # AI explanations
├── static/               # Static assets
│   └── styles.css        # Custom styles
├── requirements.txt      # Python dependencies
└── README.md             # This file

🧠 How It Works
User Input

Submit Allocation, Request, and Available matrices via web form

AI Analysis

Banker's Algorithm verifies safe/unsafe state

ML model predicts deadlock probability

Combined decision for robust results

Output

Clear safe/deadlock indication

Safe sequence (if applicable)

Probability score and explanation
