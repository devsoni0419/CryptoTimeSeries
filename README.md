# 📈 Crypto Time Series Forecasting Dashboard

A Streamlit-based time series forecasting dashboard for cryptocurrency data, designed to **demonstrate model behavior** on financial time series using **statistically correct methodology** rather than claiming trading or investment accuracy.

---

## 🚀 Project Overview

This project compares multiple time-series forecasting approaches on cryptocurrency data:

- Naive Baseline
- ARIMA
- SARIMA
- Prophet
- LSTM (Long Short-Term Memory)

The primary goal is to **understand how different models behave on financial time series**, especially when data exhibits properties close to a random walk.

---

## 🧠 Why Forecast Returns Instead of Prices

Forecasting raw prices often leads to misleading results because financial prices are typically **non-stationary** and dominated by trends.

This project forecasts **daily returns**, defined as:

Return_t = (Close_t - Close_{t-1}) / Close_{t-1}


Forecasting returns:
- Improves stationarity
- Aligns with financial modeling best practices
- Prevents false trend-following behavior

**References:**
- Forecasting: Principles and Practice (Hyndman & Athanasopoulos)  
  https://otexts.com/fpp3/stationarity.html
- Random Walk Theory (Investopedia)  
  https://www.investopedia.com/terms/r/randomwalktheory.asp

---

## 📊 Models Implemented

### 1. Naive Baseline
- Predicts the last observed return for all future steps
- Serves as a minimum benchmark
- Provides context for evaluating other models

### 2. ARIMA
- Autoregressive Integrated Moving Average
- Models short-term autocorrelation in returns
- Commonly used for stationary time series

### 3. SARIMA
- Seasonal extension of ARIMA
- Includes weekly seasonality (7-day cycle)
- Evaluates potential calendar effects

### 4. Prophet
- Additive decomposable model
- Configured with weekly seasonality
- Often converges to trend-like behavior on financial data

### 5. LSTM
- Recurrent neural network for sequence modeling
- Trained using sliding windows of returns
- Demonstrates deep learning behavior on time series data

**References:**
- Statsmodels Time Series Analysis  
  https://www.statsmodels.org/stable/tsa.html
- Prophet Documentation  
  https://facebook.github.io/prophet/
- Keras LSTM Documentation  
  https://keras.io/api/layers/recurrent_layers/lstm/

---

## 📈 Evaluation Methodology

All models are evaluated using **walk-forward validation** with a fixed forecast horizon.

### Metrics Used
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**

A model is considered useful only if it performs better than the naive baseline.

**References:**
- Forecast Accuracy Metrics  
  https://otexts.com/fpp3/forecast-accuracy.html

---

## ❗ Why Do the Results Look Similar?

It is expected for multiple forecasting models to produce similar results on financial return data because:

- Financial returns often resemble a **random walk**
- Predictive signal is weak
- Volatility dominates structure
- Deep learning models can underfit without strong features

This convergence is **not a bug**—it is an important empirical observation.

**References:**
- The Elements of Statistical Learning  
  https://www.statlearning.com/
- Efficient Market Hypothesis  
  https://www.investopedia.com/terms/e/efficientmarkethypothesis.asp

---

## ⚠️ Limitations

- No trading strategy is implemented
- No transaction costs are modeled
- No external regressors (volume, sentiment, macro data)
- Forecasts are not investment advice

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas / NumPy
- Statsmodels
- Prophet
- TensorFlow / Keras
- Plotly

---

## 📁 Project Structure

cryptotimeseries/
│
├── app/
│ └── dashboard.py
│
├── src/
│ ├── arima_model.py
│ ├── sarima_model.py
│ ├── prophet_model.py
│ └── lstm_model.py
│
├── data/
│ └── processed_stock_data.csv
│
└── requirements.txt


---

## ✅ Conclusion

This project demonstrates that **correct methodology matters more than complex models**.  
When applied properly, different forecasting techniques often converge on financial time series — a result consistent with established financial theory.

This dashboard serves as:
- A learning tool
- A portfolio project
- An empirical demonstration of time-series behavior

---

## 📜 Disclaimer

This project is for educational purposes only and does not provide financial or investment advice.


