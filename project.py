# ===========================================
# California House Price Prediction Project
# Multiple Linear Regression
# ===========================================

# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import joblib

# ===========================================
# Load Dataset
# ===========================================

df = pd.read_csv("data.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# ===========================================
# Data Cleaning
# ===========================================

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Values:", df.duplicated().sum())

print("\nRemoving missing values...")

df.dropna(inplace=True)

print("\nRemaining Missing Values")
print(df.isnull().sum())

# ===========================================
# Exploratory Data Analysis
# ===========================================

numerical_columns = [

    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "median_house_value"

]

# ===========================================
# Pairplot
# ===========================================

sns.pairplot(df[numerical_columns])
plt.show()

# ===========================================
# Correlation Matrix
# ===========================================

plt.figure(figsize=(10,8))

corr = df[numerical_columns].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")
plt.show()

# ===========================================
# Histograms
# ===========================================

df[numerical_columns].hist(figsize=(14,10), bins=20)

plt.tight_layout()
plt.show()

# ===========================================
# Boxplots
# ===========================================

plt.figure(figsize=(14,10))

for i, col in enumerate(numerical_columns,1):

    plt.subplot(3,3,i)

    sns.boxplot(y=df[col])

    plt.title(col)

plt.tight_layout()
plt.show()

# ===========================================
# Encode Categorical Variable
# ===========================================

print("\nEncoding ocean_proximity...")

df = pd.get_dummies(
    df,
    columns=["ocean_proximity"],
    drop_first=True
)

print(df.head())

# ===========================================
# Feature Selection
# ===========================================

X = df.drop("median_house_value", axis=1)

y = df["median_house_value"]

# ===========================================
# Train Test Split
# ===========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42

)

# ===========================================
# Build Model
# ===========================================

model = LinearRegression()

model.fit(X_train, y_train)

# ===========================================
# Prediction
# ===========================================

y_pred = model.predict(X_test)

# ===========================================
# Evaluation
# ===========================================

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Performance")

print("MSE :", mse)

print("RMSE:", rmse)

print("R² Score :", r2)

# ===========================================
# Feature Importance
# ===========================================

coefficients = pd.DataFrame({

    "Feature": X.columns,

    "Coefficient": model.coef_

})

coefficients = coefficients.sort_values(
    by="Coefficient",
    key=abs,
    ascending=False
)

print("\nFeature Coefficients")

print(coefficients)

# ===========================================
# Actual vs Predicted
# ===========================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.xlabel("Actual House Price")

plt.ylabel("Predicted House Price")

plt.title("Actual vs Predicted House Prices")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.show()

# ===========================================
# Predict New House
# ===========================================

new_house = pd.DataFrame({

    "longitude":[-122.23],
    "latitude":[37.88],
    "housing_median_age":[41],
    "total_rooms":[880],
    "total_bedrooms":[129],
    "population":[322],
    "households":[126],
    "median_income":[8.3252],

    "ocean_proximity_INLAND":[0],
    "ocean_proximity_ISLAND":[0],
    "ocean_proximity_NEAR BAY":[1],
    "ocean_proximity_NEAR OCEAN":[0]

})

prediction = model.predict(new_house)

print("\nPredicted House Price : $", prediction[0])

# ===========================================
# Save Model
# ===========================================

joblib.dump(model,"house_price_model.pkl")

print("\nModel Saved Successfully")