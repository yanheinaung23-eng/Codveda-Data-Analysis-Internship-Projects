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

#### 2.1. Close Prices Over Time

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

#### 2.2. Time Series Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose # imported decompose function

result = seasonal_decompose(
    df['close'],
    model='additive',
    period=30 # daily time series
)

result.plot()
plt.show()
```
![image](https://github.com/yanheinaung23-eng/Codveda-Data-Analysis-Internship-Projects/blob/b2cd9be63e31c400fe1c9cbbc539074f08583866/02_level_task_projects/02_time_series_analysis_project/images/Seasonal%20Decompose.png)

The series was decomposed into:

- Original Series
- Trend
- Seasonal Component
- Residual Component

### Findings

#### Trend

- Strong upward movement from 2014 to 2017.

#### Seasonality

- Weak seasonal behavior.
- No significant repeating monthly pattern.

#### Residuals

- Most remaining variation comes from unpredictable market fluctuations.

---

#### 2.3. Moving Average (Close Prices)

Two moving averages were calculated:

- 7-Day Moving Average
- 30-Day Moving Average

```python
df['MA7'] = df['close'].rolling(window=7).mean()

df['MA30'] = df['close'].rolling(window=30).mean()

plt.figure(figsize=(10,6))

plt.plot(df['close'], label='Close Prices')
plt.plot(df['MA7'], label='7-Day MA')
plt.plot(df['MA30'], label='30-Day MA')

plt.legend()

plt.title("Moving Average (Close Prices)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
```

![image](https://github.com/yanheinaung23-eng/Codveda-Data-Analysis-Internship-Projects/blob/b2cd9be63e31c400fe1c9cbbc539074f08583866/02_level_task_projects/02_time_series_analysis_project/images/Moving%20Average%20(Close%20Prices).png)

### Findings

Both moving averages increase steadily over time.

This confirms that:

The market trend is bullish.
The long-term trend is stable.
Short-term volatility does not change the overall upward direction.

---

## 📊 Key Findings

### Trend

The stock exhibited a sustained upward trend throughout the analysis period.

### Volatility

Frequent daily price fluctuations indicate a volatile market.

### Seasonality

Little evidence of strong recurring seasonal behavior was observed, which is typical for financial market data.

### Residuals

Residual variation suggests that many stock price movements are driven by unpredictable external factors.

### Moving Average

The moving averages effectively smooth daily fluctuations and provide a clearer view of the underlying long-term trend.

---

## 👨‍💻 Author

Yan Hein Aung

⭐ If you found this project helpful, feel free to star the repository!



