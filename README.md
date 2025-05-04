# California Housing Price Analysis and Prediction

## Project Description

This project focuses on analyzing and predicting housing prices in California using a dataset that includes various features such as location, age of the house, number of rooms, population, household size, median income, and proximity to the ocean. The objective is to build robust machine learning models to estimate the **median house value** based on these features.

---

## Project Walkthrough

### 1. Data Exploration and Preprocessing

- Loaded the California housing dataset and explored its structure and descriptive statistics.
- Handled missing values by imputing with the median of each respective column.
- Engineered new features to better capture key patterns:
  - `rooms_per_household`
  - `population_per_household`
- Applied one-hot encoding to the categorical feature `ocean_proximity`.

### 2. Feature Engineering

- Created interaction terms to capture the combined effect of multiple features.
- Standardized numerical features to have zero mean and unit variance for improved model performance.

### 3. Modeling

Developed and compared the performance of three machine learning models:

- **Linear Regression**  
  A baseline model that assumes a linear relationship between features and target variable.

- **Decision Tree Regressor**  
  A non-linear model that segments the data into branches based on feature thresholds, capturing complex decision paths.

- **Random Forest Regressor**  
  An ensemble method that averages multiple decision trees to reduce overfitting and improve generalization.

### 4. Model Evaluation

- Evaluation Metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
- Utilized cross-validation to assess model performance and ensure generalizability to unseen data.

### 5. Model Tuning

- Conducted hyperparameter tuning using Grid Search to identify the best parameter combinations for each model.
- Enhanced predictive performance and reduced the risk of overfitting.

---

## Models Used

| Model                  | Description |
|------------------------|-------------|
| **Linear Regression**  | Captures general trends with high interpretability. |
| **Decision Tree**      | Learns non-linear patterns; risk of overfitting if not controlled. |
| **Random Forest**      | Averages multiple decision trees to improve accuracy and reduce variance. |

---

## Tools and Libraries

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## Future Enhancements

- Integrate external data sources (e.g., crime rates, education quality, public services).
- Explore advanced ensemble models such as Gradient Boosting or XGBoost.
- Develop and deploy a web application using Flask or Streamlit for interactive predictions.

---


