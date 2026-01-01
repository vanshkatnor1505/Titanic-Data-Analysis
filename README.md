# 🚢 Titanic Survival Prediction using Logistic Regression

## 📌 Project Overview

This project builds a machine learning model to predict passenger survival , complete exploratory data analysis and visualization of the Titanic dataset.

The workflow covers **data cleaning, feature engineering, EDA, visualization with graphs , model training, and evaluation**, following best practices in applied data science.

---

## 🎯 Objective

To predict whether a passenger survived the Titanic disaster based on demographic and travel-related features using a **Logistic Regression** classifier.

---

## 📊 Dataset

- **Source:** Titanic Dataset (`titanic.csv`)
- **Total records:** 891
- **Target variable:** `Survived`
    - `1` → Survived
    - `0` → Did not survive

### Class Distribution

| Class | Count | Percentage |
| --- | --- | --- |
| Survived | 342 | 38% |
| Did Not Survive | 549 | 62% |

> The dataset is moderately imbalanced, so metrics beyond accuracy are required.
> 

---

## 🛠️ Data Preprocessing

### Data Cleaning

- `Age` → imputed using **median**
- `Embarked` → imputed using **mode**
- Dropped non-informative text columns:
    - `PassengerId`
    - `Name`
    - `Ticket`

### Feature Engineering

- **FamilySize** = `SibSp + Parch + 1`
- **IS_Cabin**
    - `1` → Cabin information available
    - `0` → Cabin missing

### Encoding & Scaling

- One-hot encoding applied to:
    - `Sex`
    - `Embarked`
- Numerical features scaled using **StandardScaler**
- Data split:
    - **80% training**
    - **20% testing**
    - `random_state = 42`

---

## 🤖 Model

- **Algorithm:** Logistic Regression
- **Reason for choice:**
    - Strong baseline for binary classification
    - Interpretable
    - Fast and efficient

---

## 📈 Model Performance

### Accuracy

- **Training Accuracy:** 79.78%
- **Test Accuracy:** 82.12%

> Close train–test performance indicates good generalization and no overfitting.
> 

---

### Confusion Matrix

```
[[91 14]
 [18 56]]

```

| Metric | Count |
| --- | --- |
| True Negatives | 91 |
| False Positives | 14 |
| False Negatives | 18 |
| True Positives | 56 |

---

### Classification Report

| Class | Precision | Recall | F1-score |
| --- | --- | --- | --- |
| Did Not Survive (0) | 0.83 | 0.87 | 0.85 |
| Survived (1) | 0.80 | 0.76 | 0.78 |
| **Overall Accuracy** |  |  | **0.82** |

---

## 🔍 Key Insights

- **Sex** is the strongest predictor of survival.
- **Passenger class** and **cabin availability** have significant impact.
- The model captures historical survival patterns accurately (e.g., higher survival for females and higher classes).

---

## ✅ Conclusion

The Logistic Regression model provides a **strong and interpretable baseline**, achieving 82% test accuracy with balanced precision and recall.

This makes it a reliable foundation for further improvements and experimentation.

---

## 🚀 Future Improvements

- ROC–AUC analysis
- Threshold tuning to reduce false negatives
- Cross-validation
- Feature importance analysis
- Ensemble models (Random Forest, Gradient Boosting)

---

## 🧠 Skills Demonstrated

- Data Cleaning & Imputation
- Feature Engineering
- Categorical Encoding
- Model Training & Evaluation
- Real-world ML debugging
- Interpretation of results

---

---

---

## 🔎 Exploratory Data Analysis (EDA)

Comprehensive exploratory analysis was conducted to understand survival patterns across demographic and socio-economic features.

---

### 📌 Overall Survival Statistics

- **Total passengers:** 891
- **Survived:** 342 (**38%**)
- **Did not survive:** 549 (**62%**)

This confirms a **moderately imbalanced dataset**, reinforcing the need for evaluation metrics beyond accuracy.

---

### 👩‍🦰 Survival by Sex

| Sex | Survived | Total | Survival Rate |
| --- | --- | --- | --- |
| Female | 233 | 314 | **74.2%** |
| Male | 109 | 577 | **19.6%** |

🔹 **Insight:**

Sex is the strongest individual predictor of survival. Females were significantly more likely to survive than males.

---

### 👶🧓 Survival by Age Groups

### Age < 21

- **Survived:** 82 / 180 → **45.3%**
- Males: 103 total → 48 survived (**46.6%**)
- Females: 77 total → 34 survived (**44.2%**)

### Age 21–40

- **Survived:** 205 / 563 → **36.5%**
- Males: 374 total → 62 survived (**16.6%**)
- Females: 189 total → 143 survived (**75.7%**)

### Age 41–80

- **Survived:** 55 / 148 → **37.2%**
- Males: 100 total → 18 survived (**18.0%**)
- Females: 48 total → 37 survived (**77.1%**)

🔹 **Insights:**

- Female survival rates remain high across all age groups.
- Male survival rates are consistently low regardless of age.
- Age alone is a weak predictor, but **Age + Sex interaction** provides valuable signal.

---

### 🎟️ Survival by Passenger Class (Pclass)

### Pclass 1

- Survived: 136 / 216 → **63.0%**
- Female survival: **96.8%**
- Male survival: **36.9%**

### Pclass 2

- Survived: 119 / 184 → **64.7%**
- Female survival: **92.1%**
- Male survival: **15.7%**

### Pclass 3

- Survived: 87 / 491 → **17.7%**
- Female survival: **50.0%**
- Male survival: **13.5%**

🔹 **Insights:**

- Passenger class is a strong ordinal predictor.
- Third-class males had the lowest survival probability.
- First- and second-class females had extremely high survival rates.

---

### 🌍 Survival by Embarkation Port

| Port | Survived | Total | Survival Rate |
| --- | --- | --- | --- |
| Southampton (S) | 219 | 646 | 33.9% |
| Cherbourg (C) | 93 | 168 | **55.4%** |
| Queenstown (Q) | 30 | 77 | 39.0% |

### Gender Breakdown

- Females consistently outperformed males across all embarkation points.
- Cherbourg passengers had the highest overall survival.

🔹 **Insight:**

Embarkation port adds secondary predictive value and is correlated with passenger class.

---

### 🛏️ Cabin Availability

| Cabin Status | Survived | Total | Survival Rate |
| --- | --- | --- | --- |
| Cabin Present | 136 | 204 | **66.7%** |
| Cabin Missing | 206 | 687 | 30.0% |

🔹 **Insight:**

Cabin availability strongly correlates with survival and socio-economic status.

This justified the creation of the **`IS_Cabin`** feature.

---

## 🧠 Key EDA Takeaways

- **Sex** is the single most influential feature.
- **Passenger class** and **cabin availability** provide strong socio-economic signals.
- **Age alone** is noisy but becomes informative when combined with sex.
- Feature engineering is essential to capture these interactions.
