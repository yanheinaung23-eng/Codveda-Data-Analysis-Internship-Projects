# Codveda Technologies Data Analyst Internship Projects

This repository is a portfolio of the projects completed during the **Data Analyst Internship at [Codveda Technologies](https://www.linkedin.com/company/codveda-technologies/posts/?feedView=all)**. It progresses through three difficulty levels — from data cleaning and exploratory analysis, to regression and time series modeling, to a full end-to-end machine learning pipeline — covering **5 tasks across 4 projects**.
 
Each project folder contains its own detailed `README.md` with full methodology, code snippets, visualizations, and results. This page is a map to help you jump straight to what you're interested in.
 
---
 
## 🧭 Quick Navigation
 
| Level | Task(s) | What it demonstrates | Project | Explore |
|:---:|:---:|---|---|:---:|
| 1 | 1 & 2 | Data quality checks, statistical summaries, correlation analysis, visualization | [Data Cleaning & EDA — Iris Dataset](01_level_task_projects) | [Notebook](01_level_task_projects/data_cleaning_and_EDA.ipynb) |
| 2 | 3 | Supervised ML, Linear Regression, model evaluation, model persistence | [Regression Analysis — Stock Price Prediction](02_level_task_projects/01_regression_analysis_project) | [Notebook](02_level_task_projects/01_regression_analysis_project/01_linear_regression_stock_model.ipynb) |
| 2 | 4 | Trend/seasonality decomposition, moving averages, market behavior interpretation | [Time Series Analysis — Stock Closing Prices](02_level_task_projects/02_time_series_analysis_project) | [Notebook](02_level_task_projects/02_time_series_analysis_project/02_time_series_analysis.ipynb) |
| 3 | 5 | End-to-end ML pipeline, model comparison, hyperparameter tuning, business insights | [Customer Chrun Prediction Model](https://github.com/yanheinaung23-eng/Codveda-Data-Analysis-Internship-Projects/tree/main/03_level_task_project) | [Notebook](03_level_task_project/task_5_customer_churn_prediction.ipynb) |
 
---
 
## 🌸 Task 1 & 2 — Data Cleaning & Exploratory Data Analysis (Iris Dataset)
 
<img src="https://raw.githubusercontent.com/yanheinaung23-eng/Codveda-Data-Analysis-Internship-Projects/main/01_level_task_projects/images/pairplot.png" width="600">
A foundational data-analysis project on the classic Iris dataset. The workflow inspects data structure and integrity, removes duplicate records, and calculates descriptive statistics (mean, median, mode) grouped by species. It then explores feature relationships through pairplots, scatterplots with regression lines, boxplots, and a correlation heatmap.
 
**Key finding:** petal length and petal width are strongly correlated and cleanly separate the three species, making them the most valuable features for downstream classification — while sepal measurements are far less discriminative.
 
- **Skills:** data integrity checks, `groupby` aggregations, correlation analysis, multivariate visualization
- **Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn
- **➡️ [Read the full project](01_level_task_projects)**
---
 
## 📈 Task 3 — Regression Analysis: Stock Price Prediction
 
<img src="https://raw.githubusercontent.com/yanheinaung23-eng/Codveda-Data-Analysis-Internship-Projects/main/02_level_task_projects/01_regression_analysis_project/Images/Linear%20Regression%20Actual%20vs%20Predicted.png" width="600">

A supervised machine learning project that builds a **Linear Regression model** (scikit-learn) to predict a stock's closing price from its opening price, using a dataset of 490K+ historical trading records. The full ML pipeline is covered: feature/target selection, an 80/20 train-test split, model training, evaluation, and prediction on unseen data — with the trained model saved via `joblib` for reuse.
 
**Key result:** the model achieves an R² of **99.97%** and a low RMSE of **1.65**, indicating an almost perfectly linear relationship between opening and closing price.
 
- **Skills:** regression modeling, model evaluation (R², MSE, MAE, RMSE), model serialization
- **Tools:** Python, Pandas, NumPy, Matplotlib, Scikit-learn, Joblib
- **➡️ [Read the full project](02_level_task_projects/01_regression_analysis_project)**
---
 
## 📊 Task 4 — Time Series Analysis: Stock Closing Prices
 
![Alt image](https://github.com/yanheinaung23-eng/Codveda-Data-Analysis-Internship-Projects/blob/c130156a0113d443ad89f58b8358c8bb22d2f51d/02_level_task_projects/02_time_series_analysis_project/images/Seasonal%20Decompose.png)

An exploratory time series analysis of historical stock prices (2014–2017). The series is decomposed into **trend, seasonal, and residual components**, and 7-day/30-day moving averages are applied to smooth out daily volatility and expose the underlying long-term direction.
 
**Key finding:** a sustained upward trend over the period with high short-term volatility and only weak seasonality — typical of financial market data — with most residual variation driven by unpredictable external factors.
 
- **Skills:** time series decomposition, trend/seasonality analysis, moving-average smoothing
- **Tools:** Python, Pandas, Matplotlib, Statsmodels
- **➡️ [Read the full project](02_level_task_projects/02_time_series_analysis_project)**
---
 
## 📉 Task 5 — Customer Churn Prediction
 
<img src="https://raw.githubusercontent.com/yanheinaung23-eng/Codveda-Data-Analysis-Internship-Projects/main/03_level_task_project/Images/feature_importance.png" width="600">

The capstone project: an end-to-end classification pipeline predicting which telecom customers are likely to churn. Three models:   **Logistic Regression, Decision Tree, and Random Forest**  — are built on an identical preprocessing pipeline (`ColumnTransformer` + `Pipeline`) and compared fairly on the same metrics. The best-performing model is then tuned with `GridSearchCV`, optimized for F1-score to balance precision and recall, and its feature importances are used to generate concrete business recommendations.
 
**Key result:** the tuned Random Forest reaches **93.3% accuracy** and **73.1% recall** (up from 55.1% before tuning), correctly catching significantly more real churners — the metric that matters most for a retention use case. The top churn drivers were customer service call frequency, day-time usage/charges, and international plan enrollment.
 
- **Skills:** ML pipelines, model comparison, hyperparameter tuning, feature importance, translating model output into business insight
- **Tools:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib
- **➡️ [Read the full project](03_level_task_project)**
---
 
## 🛠️ Overall Tech Stack
 
| Category | Tools |
|---|---|
| Languages | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (Linear Regression, Logistic Regression, Decision Tree, Random Forest, GridSearchCV) |
| Time Series | Statsmodels |
| Model Persistence | Joblib |
| Environment | Jupyter Notebook, VS Code |
 
---
 
## 📁 Repository Structure
 
```
Codveda-Data-Analysis-Internship-Projects/
│
├── 01_level_task_projects/                     # Task 1 & 2 — Data Cleaning & EDA
│   ├── data_cleaning_and_EDA.ipynb
│   ├── 1) iris.csv
│   ├── images/
│   └── README.md
│
├── 02_level_task_projects/
│   ├── 01_regression_analysis_project/         # Task 3 — Regression Analysis
│   │   ├── 01_linear_regression_stock_model.ipynb
│   │   ├── data/
│   │   ├── Images/
│   │   └── README.md
│   │
│   └── 02_time_series_analysis_project/        # Task 4 — Time Series Analysis
│       ├── 02_time_series_analysis.ipynb
│       ├── data/
│       ├── images/
│       └── README.md
│
├── 03_level_task_project/                      # Task 5 — Customer Churn Prediction
│   ├── task_5_customer_churn_prediction.ipynb
│   ├── dataset/
│   ├── Images/
│   └── README.md
│
└── README.md                                    # ← You are here
```
 
---
 
## 👤 About the Author
 
**Yan Hein Aung** — Data Analyst Intern at Codveda Technologies
 
🔗 [GitHub Profile](https://github.com/yanheinaung23-eng)
 
⭐ If you found this portfolio helpful, feel free to star the repository!
