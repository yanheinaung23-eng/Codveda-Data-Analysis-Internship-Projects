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

#### Close Prices Over Time

A line chart was created to observe stock price movement over time.
```python
plt.figure(figsize=(10,6))

plt.plot(df["close"])
plt.xlabel("Date")
plt.ylabel("Close Prices")
plt.title("Close Prices Over Time")

plt.show()
```
![image](https://github.com/yanheinaung23-eng/Codveda-Data-Analysis-Internship-Projects/blob/c5669e2a671b9bda815a34c2adb190d7777a765a/02_level_task_projects/02_time_series_analysis_project/images/Close%20Prices%20Over%20Time.png)

### Insights

✅ Overall Upward Trend

The closing price generally increases from around 1,100 in early 2014 to approximately 1,800–2,000 by late 2017.
This suggests the asset experienced sustained long-term growth.

✅ High Volatility

There are frequent short-term fluctuations throughout the period.
Volatility becomes more pronounced after 2016, indicating larger daily price swings.

✅ Market Correction

Near the end of 2017, prices decline from their peak.
This likely represents a market correction following a strong upward movement.

---






