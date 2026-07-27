# 📈 Stock Price Prediction using Linear Regression

## 📌 Project Overview

This project demonstrates the implementation of a **Machine Learning Regression Model** using **Scikit-learn** to predict stock closing prices based on historical market data.

Please check my full python workflow [here](https://github.com/yanheinaung23-eng/Codveda-Data-Analysis-Internship-Projects/blob/e08091391a181faadb18e60f66f43a684c7da938/02_level_task_projects/01_regression_analysis_project/01_linear_regression_stock_model.ipynb). 👈

The workflow covers the complete machine learning pipeline, including data preprocessing, train-test splitting, model training, evaluation, visualization, and prediction of unseen data.

---

## 🎯 Objectives

- Build a Linear Regression model for stock price prediction.
- Train and evaluate the regression model.
- Visualize model performance.
- Predict future closing prices from new input data.
- Save and reuse the trained model.

---

## 📊 Dataset

The [dataset](https://github.com/yanheinaung23-eng/Codveda-Data-Analysis-Internship-Projects/tree/59a020b2d5b67c129d63fa9f85aa374118f3a5ae/02_level_task_projects/01_regression_analysis_project/data) contains 490K+ rows of historical stock market information including:

- Date
- Open Price
- High Price
- Low Price
- Close Price
- Volume

Target Variable:

- **Close Price**

Feature Used:

- **Open Price**

---

## 🛠 Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

---

## 📂 Project Workflow

### 1. Data Preparation

- Import libraries and Load dataset
- Inspect the data
- Handle missing values
- Select features and target
```python
X = df[["open"]]
y = df["close"] 
```
- Split training and testing data (80/20)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
---

### 2. Model Training

Linear Regression was trained using Scikit-learn.
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```
---

### 3. Model Prediction

```python
y_pred = model.predict(X_test)
```

---
## 📊 Model Performance

The Linear Regression model was evaluated using multiple regression metrics to assess its predictive accuracy.

### R-squared and Mean Squared Error

```python
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R-squared:", r2)
print("Mean Squared Error:", mse)
```
*R-squared: 0.9997383343436389*

*Mean Squared Error: 2.726082368531974*

| Metric | Value | Interpretation |
|---------|-------|----------------|
| **R-squared Score** | **0.999738** | The model explains **99.97%** of the variance in the closing prices, indicating an excellent fit to the data. |
| **Mean Squared Error (MSE)** | **2.7261** | The average squared prediction error is very small, suggesting high prediction accuracy. |

---

### Coefficient and Intercept
```python
coefficient = model.coef_[0]
intercept = model.intercept_

print("Coefficient (Slope):", coefficient)  
print("Intercept:", intercept)
```
*Coefficient (Slope): 0.999874498139709*

*Intercept: 0.026677540332286753*

| Metric | Value | Interpretation |
|---------|-------|----------------|
| **Slope (Coefficient)** | **0.999874** | A one-unit increase in the opening price results in an approximate **0.9999-unit increase** in the predicted closing price, indicating an almost one-to-one linear relationship. |
| **Intercept** | **0.026678** | When the opening price is zero, the predicted closing price is approximately **0.027**. Although not practically meaningful for stock prices, the intercept is required to define the regression equation. |

### MAE and RMSE
```python
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
```
*MAE: 0.7691860843468409*

*RMSE: 1.6510852093492856*

| Metric | Value | Interpretation |
|---------|-------|----------------|
| **Mean Absolute Error (MAE)** | **0.7692** | The average absolute prediction error is less than **1 price unit**, demonstrating precise predictions. |
| **Root Mean Squared Error (RMSE)** | **1.6511** | On average, the model's predictions differ from the actual closing price by approximately **1.65 price units**. |

---

## 📈 Regression Equation

The trained Linear Regression model can be expressed as:

```text
Predicted Close Price = (0.999874 × Open Price) + 0.026678
```

This equation indicates an almost perfect positive linear relationship between the opening and closing prices.

**Visualization: Actual vs Predicted**
```python
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, label="Actual Data")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regression Line")

plt.xlabel("Open Price")
plt.ylabel("Close Price")
plt.title("Linear Regression")
plt.legend()
plt.show()
```
![Alt image](https://github.com/yanheinaung23-eng/Codveda-Data-Analysis-Internship-Projects/blob/35d86da85da1cb04b96e70854febd1ff14d4adf7/02_level_task_projects/01_regression_analysis_project/Images/Linear%20Regression%20Actual%20vs%20Predicted.png)

---

## Testing with few new data
```python
new_data = [
    [120.321],
    [130.234],
    [140.122],
    [150.903]
]

test_pred = model.predict(new_data)
print(test_pred)
```
*120.33257703*

*130.24433293*

*140.13109197*   

*150.91073893*

---

## Finally, Saving the Model!
```python
import joblib
joblib.dump(model, "linear_regression_stock_model.pkl")
```
---

We can use this model with following steps:
```python
import joblib
model = joblib.load("linear_regression_stock_model.pkl")
```

----

## 📊 Results

The Linear Regression model successfully learned the relationship between stock opening prices and closing prices.

The evaluation metrics and visualizations indicate that the model captures the overall price trend while maintaining relatively small prediction errors.

---
## 👨‍💻 Author

Yan Hein Aung

⭐ If you found this project helpful, feel free to star the repository!

