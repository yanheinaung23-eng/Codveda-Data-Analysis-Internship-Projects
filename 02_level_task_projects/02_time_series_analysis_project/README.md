# 📊 Time Series Analysis of Stock Closing Prices

## 📌 Project Overview

This project performs an exploratory **Time Series Analysis (TSA)** on historical stock prices using Python.

The analysis focuses on identifying long-term trends, seasonal patterns, and short-term fluctuations through visualization, seasonal decomposition, and moving average smoothing.

---

## 🎯 Objectives

- Plot time-series data.
- Identify trends and market behavior.
- Decompose the series into:
  - Trend
  - Seasonality
  - Residual
- Apply moving average smoothing.
- Interpret stock price movements over time.

---

## 📊 Dataset

Historical stock market dataset containing:

- Date
- Open
- High
- Low
- Close
- Volume

Time Period:

2014 – 2017

---

## 🛠 Technologies

- Python
- Pandas
- Matplotlib
- Statsmodels

---

## 📂 Project Workflow

### 1. Data Inspecting and Preparation

- Import libraries and Load dataset
- Inspect the data
- Handle missing values
- Convert Date column to datetime
```python
df["date"] = pd.to_datetime(df["date"])
```
- Sort the Date column
```python
df = df.sort_values(by="date")
```
- Set Date as index
```python
df = df.set_index("date")
```
---

### 2. Time Series Visualization

A line chart was created to observe stock price movement over time.
```python
plt.figure(figsize=(10,6))

plt.plot(df["close"])
plt.xlabel("Date")
plt.ylabel("Close Prices")
plt.title("Close Prices Over Time")

plt.show()
```





