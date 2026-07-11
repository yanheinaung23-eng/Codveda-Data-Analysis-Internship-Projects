# 🌸 Iris Dataset Data Cleaning & Exploratory Data Analysis

---

## 📌 Project Overview

This project demonstrates a complete data cleaning and exploratory data analysis (EDA) workflow using the Iris Dataset. The objective is to inspect data quality, remove duplicate records, generate descriptive statistics, and uncover patterns through visualizations to uncover **insights** and **correlation mathematical patterns**.

Full python code end to end: [Click here.](https://github.com/yanheinaung23-eng/Codveda-Data-Analysis-Internship-Projects/blob/f590f97d0c1c5d0223c40e5fa5011e221f7fe45b/01_level_task_projects/data_cleaning_and_EDA.ipynb)

---

## 🎯 Objectives

1. Inspect and clean the dataset
2. EDA
    * Perform statistical analysis
    * Explore relationships between measurements
    * Distributions and correlations
3. Insights

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
## 📊 Exploratory Data Analysis

### Statistical Analysis

Calculated the following metrics grouped by species:
* Mean
* Median
* Mode
These statistics were compared using bar charts to understand species-level differences.

### Means calculation
```python
print("Mean values across the dataset by species: ")
df_mean = df.groupby("species").mean()
display(df_mean)
```
| Species | Avg. Sepal Length | Avg. Sepal Width | Avg. Petal Length | Avg. Petal Width |
|:---------|------------------:|-----------------:|------------------:|-----------------:|
| Setosa | 5.010417 | 3.431250 | 1.462500 | 0.250000 |
| Versicolor | 5.936000 | 2.770000 | 4.260000 | 1.326000 |
| Virginica | 6.604082 | 2.979592 | 5.561224 | 2.028571 |

---
### Median calculation
```python
print("Median values across the dataset by species: ")
df_median = df.groupby('species').median()
display(df_median)
```
| Species | Median Sepal Length | Median Sepal Width | Median Petal Length | Median Petal Width |
|:---------|--------------------:|-------------------:|--------------------:|-------------------:|
| Setosa | 5.0 | 3.4 | 1.50 | 0.2 |
| Versicolor | 5.9 | 2.8 | 4.35 | 1.3 |
| Virginica | 6.5 | 3.0 | 5.60 | 2.0 |

---
### Mode calculation
```python
print("Mode values across the dataset by species: ")
df_mode = df.groupby("species").agg(lambda x: x.mode().iloc[0])
display(df_mode)
```
*Why use iloc: The .mode() function returns a series because of the potential for ties. Adding .iloc[0] ensures that if a specific feature has a tie within a species, it cleanly selects the first one so your final output remains a neat, easy-to-read table.*

| Species | Mode Sepal Length | Mode Sepal Width | Mode Petal Length | Mode Petal Width |
|:---------|------------------:|-----------------:|------------------:|-----------------:|
| Setosa | 5.0 | 3.4 | 1.4 | 0.2 |
| Versicolor | 5.5 | 3.0 | 4.5 | 1.3 |
| Virginica | 6.3 | 3.0 | 5.1 | 1.8 |

### Mean, Median, Mode Visualization
![alt image](https://github.com/yanheinaung23-eng/Codveda-Data-Analysis-Internship-Projects/blob/7752a3cad1f4c45655b5a3de544436a6d9f05ee2/01_level_task_projects/images/mean%20median%20mode.png)

---
### Explore relationships between measurements
```python
sns.pairplot(data= df, hue="species", palette="Set2" )
```
![alt image](https://github.com/yanheinaung23-eng/Codveda-Data-Analysis-Internship-Projects/blob/06c930ad245d3e9c9deaafb8755e321b1c1fde6e/01_level_task_projects/images/pairplot.png)

### petal_length vs petal_width correlation
```python
plt.figure(figsize=(6,5))
sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species")
plt.title("Petal length vs Petal width Scatterplot")
sns.regplot(data=df,
            x="petal_length",
            y="petal_width", 
            scatter=False,
            color="Red",
            line_kws={"linestyle" :"--"}  
            )
plt.show()
```
![alt image](https://github.com/yanheinaung23-eng/Codveda-Data-Analysis-Internship-Projects/blob/3f03c196cc200ed9003571ef8c65eca399c2f56d/01_level_task_projects/images/petal_length%20vs%20petal_width%20scatterplot.png)

---
### sepal_length vs sepal_width correlation
```python
plt.figure(figsize=(6,5))
sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species")
plt.title("Sepal length vs Sepal width Scatterplot")
sns.regplot(data=df,
            x= "sepal_length",
            y="sepal_width",
            scatter=False,
            line_kws= {"linestyle": "--"},
            color = "red" )
plt.show()
```
![alt image]()



