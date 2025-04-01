AI-Powered Deadlock Detection System

📌 Project Overview

This project is an AI-powered Deadlock Detection System that helps in identifying whether a deadlock situation has occurred in a system based on the allocation, request, and available resource matrices. The system utilizes Machine Learning to predict deadlocks and provides a user-friendly web interface for users to input resource details and get real-time results.

🎯 Objective

The main goal of this project is to:

Detect deadlocks using AI models trained on resource allocation datasets.

Provide a simple and intuitive web interface for users to analyze deadlock scenarios.

Help users understand what deadlocks are, why they occur, and how to prevent them through informative web pages.

🚀 Features

✔ AI-based deadlock prediction using machine learning.✔ User-friendly web interface for inputting resource matrices.✔ Multiple webpages explaining deadlocks, their causes, and prevention techniques.✔ Real-time detection of deadlock scenarios.✔ Visual representation of the resource allocation system.

🛠 Tech Stack

This project is built using:

Python - Core programming language for AI & logic implementation.

Flask - Web framework for creating the user interface.

HTML, CSS - Frontend development for the web UI.

Jinja2 - Templating engine for rendering dynamic content.

Pandas & NumPy - For handling dataset and matrix computations.

Scikit-learn - For training the AI model to predict deadlock occurrences.

📂 Project Structure

AI_POWERED_DEADLOCK/
│── app.py           # Main Flask application
│── model.py         # Machine Learning model for deadlock prediction
│── templates/
│   │── index.html    # Main webpage (Home)
│   │── predictor.html # AI Predictor interface
│   │── deadlock_info.html # Deadlock explanation page
│── static/
│   │── styles.css    # CSS for UI design
│── dataset.csv      # Dataset used for training
│── README.md        # Documentation
│── requirements.txt # Dependencies

🏗️ How to Run the Project

1️⃣ Clone the repository:

 git clone https://github.com/codewithadi/AI_POWERED_DEADLOCK.git
 cd AI_POWERED_DEADLOCK

2️⃣ Create and activate a virtual environment (optional but recommended):

 python -m venv venv
 source venv/bin/activate  # For macOS/Linux
 venv\Scripts\activate     # For Windows

3️⃣ Install dependencies:

 pip install -r requirements.txt

4️⃣ Run the Flask app:

 python app.py

5️⃣ Open the browser and visit:

 http://127.0.0.1:5000/

🧠 How It Works

User Inputs: The user provides resource allocation details (allocated, requested, and available resources) in the web interface.

Preprocessing: The input data is converted into a format suitable for the AI model.

AI Model Prediction: The trained model predicts whether a deadlock will occur.

Result Display: The result is displayed on the UI, showing whether the system is in a deadlock state.

🔥 Future Enhancements

📊 Graphical visualization of resource allocation.

🛡 Deadlock prevention mechanisms based on AI recommendations.

🌐 Deploying the project online for wider accessibility.

📡 Live data integration for real-time resource monitoring.

🤝 Contributing

Want to contribute? Feel free to fork this project and submit a pull request. Contributions are always welcome! 😊

📜 License

This project is open-source and available under the MIT License.

🎉 Happy Coding & Enjoy AI-powered Deadlock Detection
