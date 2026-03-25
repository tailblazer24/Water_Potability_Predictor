# Water Potability Prediction

A machine learning project to predict whether water is safe to drink based on chemical parameters. The project includes:

- Data cleaning and exploratory data analysis (EDA)
- Model training (Logistic Regression, Random Forest) with class imbalance handling
- Feature importance analysis
- Interactive Streamlit dashboard for real-time predictions

## Files

- `app.py`: Streamlit app for prediction
- `water_quality_model.pkl`: Trained Random Forest model
- `scaler.pkl`: StandardScaler for input scaling

## How to Run

1. Clone this repo
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn streamlit`
3. Run the Streamlit app: `streamlit run app.py`

## Results

The Random Forest model achieved an ROC‑AUC of ~0.75. Feature importance shows that chloramines, pH, and solids are the most influential parameters.

<img width="1322" height="860" alt="image" src="https://github.com/user-attachments/assets/7e065505-774c-4a9b-bc3b-255ef3530083" />

