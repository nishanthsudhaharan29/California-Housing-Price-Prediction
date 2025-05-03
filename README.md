# 🏠 California Housing Price Analysis and Prediction

## 📘 Project Description

This project focuses on analyzing and predicting housing prices in California using a dataset that includes various features such as location, age of the house, number of rooms, population, household, median income, and proximity to the ocean. The objective is to build robust machine learning models to estimate the **median house value** based on these features.

---

## 🔍 Walkthrough

### 1. 📊 Data Exploration and Preprocessing

- Loaded the California housing dataset and explored its structure and descriptive statistics.
- Handled missing values by filling them with the median of the respective column.
- Engineered new features:
  - `rooms_per_household`
  - `population_per_household`
- Encoded the categorical feature `ocean_proximity` using **one-hot encoding**.

### 2. 🛠️ Feature Engineering

- Created interaction terms to capture combined feature effects.
- Standardized features to have **zero mean** and **unit variance**, improving model performance.

### 3. 🤖 Modeling

Three models were developed and compared:

- **Linear Regression**  
  A straightforward model assuming a linear relationship between features and house prices.

- **Decision Tree Regressor**  
  A non-linear model that splits data into decision nodes based on feature thresholds, capturing more complex patterns.

- **Random Forest Regressor**  
  An ensemble of decision trees providing improved accuracy by averaging multiple predictions and reducing overfitting.

### 4. 📈 Model Evaluation

- Performance Metrics:
  - **Mean Absolute Error (MAE)**
  - **Root Mean Squared Error (RMSE)**
- Employed **cross-validation** to ensure models generalize well to unseen data.

### 5. 🔧 Model Tuning

- Performed **hyperparameter tuning** using **grid search** to optimize model performance and prevent overfitting.

---

## 🧠 Models Used

| Model                  | Description |
|------------------------|-------------|
| **Linear Regression**  | Captures basic trends and relationships; fast and interpretable. |
| **Decision Tree**      | Learns non-linear boundaries; may overfit without pruning. |
| **Random Forest**      | Averages multiple trees for better generalization and accuracy. |

---

## 🛠️ Tools & Libraries

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 🚀 Future Enhancements

- Incorporate more external data (e.g., crime rates, school quality).
- Explore other ensemble techniques like Gradient Boosting or XGBoost.
- Deploy as a web app using Flask or Streamlit.

---

## 📂 Repository Structure

