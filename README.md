# 📊 Walmart Sales Forecasting Analysis

This project presents a comprehensive sales forecasting and visualization system for Walmart using time series analysis and machine learning. It features an interactive **Streamlit dashboard** that enables users to explore data patterns, evaluate model performance, and forecast weekly sales using ARIMA, Exponential Smoothing, and Random Forest.

---

## 🔍 Key Features

- 📈 Time Series Analysis using ARIMA and Holt-Winters models
- 🌐 Interactive dashboard built with Streamlit
- 📊 Exploratory Data Analysis (EDA) and sales visualizations
- 🧮 Feature engineering and correlation heatmaps
- 🎯 Random Forest Regressor for supervised prediction
- 📅 Holiday impact analysis on sales patterns

---

## 🗂️ Project Structure

```
walmart-sales-forecasting/
│
├── filename.py          # Streamlit dashboard app
├── train.csv            # Training sales data
├── test.csv             # Test dataset
├── stores.csv           # Store metadata
├── features.csv         # Additional features (CPI, Unemployment, etc.)
├── README.md            # Project documentation
└── requirements.txt     # List of dependencies
```

---

## ⚙️ Technologies Used

- **Python**  
- **Libraries**:  
  - `pandas`, `numpy` for data handling  
  - `matplotlib`, `seaborn` for visualization  
  - `statsmodels`, `pmdarima` for time series modeling  
  - `scikit-learn` for Random Forest  
  - `streamlit` for building the interactive web dashboard

---

## 🧪 Models Used

- **ARIMA (AutoRegressive Integrated Moving Average)**  
- **Holt-Winters Exponential Smoothing**  
- **Random Forest Regressor**  
- Evaluation metrics include WMAE (Weighted Mean Absolute Error)

---

## 📊 Dashboard Views

- **Data Overview**: Dataset stats, sample view, sales matrix  
- **Exploratory Analysis**: Trends, store/department sales, holiday effects  
- **Feature Engineering**: Correlation heatmaps and feature importances  
- **Modeling**: Train/test Random Forest, WMAE metrics, prediction plots  
- **Time Series**: Decomposition, stationarity checks, and forecasting  

---

## 🚀 How to Run the App

1. Clone this repo  
2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app  
   ```bash
   streamlit run filename.py
   ```

---

## 📈 Results and Insights

- Identified key temporal patterns (monthly, weekly trends)
- Evaluated the influence of holidays on sales
- Built forecasting models that outperform naive benchmarks
- Delivered visual tools for business insight and planning

---

## 📌 Future Enhancements

- Deploy as a hosted web app using Streamlit Cloud or Heroku  
- Include advanced ML models like LSTM for non-linear time series  
- Add real-time data updating and monitoring  

---



## 🙏 Acknowledgments

- Walmart Sales Forecasting Kaggle Dataset  
- Python and Streamlit open-source communities  
