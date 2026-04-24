# 🩺 AI Health Diagnostician

An advanced, interactive Disease Prediction web application built with Machine Learning. This application allows users to input their current symptoms and uses a trained Random Forest model to predict the most likely underlying disease.

## 🌟 Features
- **Machine Learning Core**: Uses an optimized Random Forest Classifier trained on a comprehensive health dataset.
- **Dynamic User Interface**: Built with Streamlit, featuring a modern, dark-mode, glassmorphic design.
- **Diagnostic Analytics**: Displays a confidence chart showing the probabilities of the top 3 most likely conditions.
- **Actionable Advice**: Provides clear descriptions of the predicted disease along with recommended medical precautions.
- **Severity Alerts**: Automatically warns the user to consult a doctor if highly severe symptoms are selected.

## 🛠️ Technology Stack
- **Frontend/Backend**: Python, Streamlit
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn

## 🚀 How to Run Locally

1. Ensure you have Python installed.
2. Clone this repository or download the files.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   python -m streamlit run app.py
   ```

## ⚠️ Disclaimer
*This application is for educational and demonstrative purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider with any questions you may have regarding a medical condition.*
