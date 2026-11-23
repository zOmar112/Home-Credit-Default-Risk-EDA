# Home Credit Default Risk — Exploratory Data Analysis & Feature Engineering

## 📌 Project Overview
This project focuses on the **Home Credit Default Risk** dataset, aiming to analyze historical loan application data to uncover key indicators of client repayment difficulties (`TARGET = 1`) versus timely repayments (`TARGET = 0`). 

The notebook covers thorough data auditing, domain-aware cleaning of hidden anomalies, feature engineering, and correlation analysis to prepare the dataset for gradient boosting models.

---

## 🛠️ Key Steps & Methodology

### 1. Data Cleaning & Anomaly Handling
* **Hidden Missing Values (`XNA`):** Identified `'XNA'` entries in `CODE_GENDER` and `ORGANIZATION_TYPE` as disguised missing values and converted them to `NaN`[cite: 4].
* **Employment Day Anomaly (`365243`):** Detected the placeholder value `365243` in `DAYS_EMPLOYED` (representing unemployed or retired applicants) and replaced it with `NaN` to prevent distribution distortion[cite: 4].

### 2. Feature Engineering
* **Missingness & Domain Indicators:** Created binary flags (`IS_UNEMPLOYED_OR_RETIRED` and `HAS_CAR`) prior to imputation to preserve informative missingness patterns[cite: 4].
* **Time-Scale Normalization:** Converted negative day counts (`DAYS_BIRTH` and `DAYS_EMPLOYED`) into positive, interpretable year-based features (`AGE_YEARS` and `YEARS_EMPLOYED`), dropping the raw columns afterward[cite: 4].

### 3. Exploratory Data Analysis (EDA)
* **Class Imbalance Assessment:** Evaluated the target distribution, identifying a significant class imbalance (~8.07% default rate)[cite: 4].
* **Correlation & Feature Importance:** Computed feature correlations with the `TARGET` variable and visualized the top 13 most influential features (such as `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`, and `AGE_YEARS`) using a correlation heatmap[cite: 4].

---

## 💻 Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Modeling & Preprocessing (Setup):** Scikit-learn (`HistGradientBoostingClassifier`, `ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`)

---

## 📂 Dataset Source
Due to file size constraints, the raw CSV files are not hosted in this repository. You can download the original dataset from Kaggle:
* [Home Credit Default Risk Dataset on Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data)
