# 📊 Nifty 50 & BankNifty Stock Predictor  

A machine learning-based predictive system to analyze and forecast the movement of **Nifty 50** and **BankNifty** indices.  
The project focuses on **trend prediction (Up/Down)** and **30-day forecasting**, with an interactive dashboard for visualization.  

This work was developed as part of an internship under **PARWORKS Innovations LLP**, showcasing the practical application of **AI in Financial Technology (FinTech)**.  

---

## ✨ Key Highlights  

- 📈 **Trend Prediction** → Forecasts Up/Down movements for Nifty 50 and BankNifty.  
- 📊 **30-Day Forecasting** → Uses ML models to project future index movements.  
- 🖥️ **Interactive Dashboard** → Streamlit app with graphs & predictions.  
- ⚙️ **Modular Code** → Clean separation of preprocessing, modeling, and visualization.  
- 🔄 **Extendable** → Can be adapted to other stock indices or financial datasets.  

---

## 🛠️ Tech Stack  

- **Python 3.10+**  
- **Pandas, NumPy** → Data preprocessing  
- **Scikit-learn** → Machine Learning (Random Forest, Logistic Regression)  
- **Matplotlib, Seaborn** → Data visualization  
- **Streamlit** → Web app deployment  

---

## 📂 Project Structure  

nifty_banknifty_predictor/
│── deploy/
│ └── app_streamlit.py # Streamlit dashboard
│
│── src/
│ ├── preprocessing.py # Data preprocessing functions
│ ├── model.py # Machine learning models
│ ├── visualization.py # Graphs & plots
│ └── data_handler.py # Data loading & cleaning
│
│── README.md # Project overview
│── requirements.txt # Dependencies

yaml
Copy code

---

## 🚀 How to Run  

1. **Clone or Extract Project**  
   - If using GitHub:  
     ```bash
     git clone https://github.com/abhishekparmar2005/nifty_banknifty_predictor.git
     cd nifty_banknifty_predictor
     ```
   - If using ZIP:  
     - Extract the ZIP file.  
     - Open the project folder in VS Code or terminal.  

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
Run the Application

bash
Copy code
streamlit run deploy/app_streamlit.py
Open in browser → http://localhost:8501

🎯 Purpose & Outcomes
✔️ Provides insights into market behavior with ML-driven predictions.
✔️ Offers a visual, user-friendly tool for analyzing financial trends.
✔️ Demonstrates end-to-end ML pipeline:

Data preprocessing

Model training & evaluation

Visualization

Deployment with Streamlit

🏆 Internship Contribution
This project was successfully completed during my internship at PARWORKS Innovations LLP.
It highlights practical industry-level exposure in Data Science & Financial Technology.
