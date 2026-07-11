# 🌸 Iris Dataset Data Cleaning & Exploratory Data Analysis

---

## 📌 Project Overview

This project demonstrates a complete data cleaning and exploratory data analysis (EDA) workflow using the Iris Dataset. The objective is to inspect data quality, remove duplicate records, generate descriptive statistics, and uncover patterns through visualizations to uncover **insights** and **correlation mathematical patterns**.

---

## 🎯 Objectives

1. Inspect and clean the dataset
2. EDA
    * Perform statistical analysis by species
    * Explore relationships between flower measurements
    * Distributions and correlations

---

## 🛠️ Tools & Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Visual Studio Code

---

## 📂 Dataset

The Iris Dataset contains measurements of three iris flower species.

### Features

| Column | Description |
|---------|-------------|
| Sepal Length | Length of sepal (cm) |
| Sepal Width | Width of sepal (cm) |
| Petal Length | Length of petal (cm) |
| Petal Width | Width of petal (cm) |
| Species | Iris flower species |

Species:

- Setosa
- Versicolor
- Virginica

---
## 📁 Project Structure

```
01_level_task_projects/
│
├── data_cleaning_and_EDA.ipynb
├── 1) iris.csv
├── README.md
└── images/
    ├── mean median mode.png
    ├── petal_length vs petal_width scatterplot.png 
    ├── sepal_length vs sepal_width scatterplot.png
    ├── pairplot.png
    ├── boxplot.png
    └── correlation heatmap.png
```
---

## 🔎 1. Inspect and clean the dataset

### Structure overview
   * Used ```df.shape```,```df.head()```, ```df.tail()```, ```df.sample(5)``` to confirm **no structure errors**.

### Integrity Check
   * Used ```df.info()```, ```df.isnull().sum()``` to verify there is **no missing values**.

### Statistical calculations
   * Used ```df.describe()``` to calculate **means, standard deviation, min and max, quartiles of 25%, 50%, 75%**.

### Remove duplicates

* Checked for duplicate rows using:

```python
df.duplicated().sum()
```

* Removed duplicate records to ensure data integrity:

```python
df.drop_duplicates(inplace=True)
```





