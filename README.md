# 🏠 California House Price Prediction

## 📌 Project Overview

This project predicts California house prices using Multiple Linear Regression. The model is trained on housing data containing information such as median income, house age, total rooms, population, and geographical location.

The goal is to understand how different features influence house prices and build a machine learning model capable of predicting the median house value.

---

## 📂 Dataset

The dataset contains information about California housing districts.

### Features

- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Median Income
- Ocean Proximity

### Target Variable

- Median House Value

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 📊 Exploratory Data Analysis (EDA)

The following analyses were performed:

- Data Cleaning
- Missing Value Handling
- Statistical Summary
- Correlation Matrix
- Histograms
- Pair Plots
- Feature Distribution Analysis

---

## 🤖 Machine Learning Model

Algorithm Used:

- Multiple Linear Regression

### Steps

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Selection
5. Train-Test Split
6. Model Training
7. Prediction
8. Model Evaluation

---

## 📈 Model Performance

| Metric | Score |
|---------|--------|
| RMSE | 69297.72 |
| R² Score | 0.6488 |

---

## 📁 Project Structure

```
California-House-Price-Prediction/
│
├── data.csv
├── House_Price_Prediction.ipynb
├── requirements.txt
├── README.md
├── model.pkl
└── images/
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/California-House-Price-Prediction.git
```

Go to the project directory

```bash
cd California-House-Price-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Jupyter Notebook

```bash
jupyter notebook
```

---

## 📌 Results

The model successfully predicts California house prices with an R² score of approximately **0.65**, demonstrating a moderate predictive performance.

---

## 🚀 Future Improvements

- Feature Engineering
- Hyperparameter Tuning
- Random Forest Regression
- XGBoost Regression
- Model Deployment using Streamlit or Flask

---

## 👩‍💻 Author

**Siya Saini**

B.Tech Computer Engineering Student

Machine Learning Enthusiast
