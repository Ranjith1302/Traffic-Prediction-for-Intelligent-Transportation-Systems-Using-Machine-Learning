# Traffic-Prediction-for-Intelligent-Transportation-Systems-Using-Machine-Learning
Developed a machine learning model to predict traffic flow patterns using historical data and time-based features for improved transporta- tion planning. • Applied data preprocessing techniques and trained regression models to generate accurate, real-time traffic predictions.


## Project Overview

This project aims to predict traffic flow based on various features like weather, temperature, zone, and day. The model will help traffic management systems predict traffic patterns and optimize routes for better traffic flow. The models are evaluated using **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **Root Mean Squared Error (RMSE)**, with a focus on selecting the best-performing model.

## Dataset

The dataset used in this project contains traffic data over a period of time, including features like:

- **Day**: Day of the week (e.g., Monday, Tuesday, etc.)
- **Date**: Date in integer form
- **CodedDay**: Encoded day number
- **Zone**: Geographical zone identifier
- **Weather**: Encoded weather condition
- **Temperature**: Temperature in Celsius
- **Traffic**: Traffic flow or number of vehicles in a given time period

The dataset has 1439 rows and 7 columns.

## Models Used

The following machine learning models were used for traffic prediction:

1. **Linear Regression**
2. **Lasso Regression**
3. **Ridge Regression**
4. **Decision Tree Regressor**
5. **Random Forest Regressor**
6. **XGBoost Regressor**
7. **Support Vector Regressor (SVR)**

## Evaluation Metrics

The models were evaluated using the following metrics:

- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in the predictions.
- **Mean Squared Error (MSE)**: Measures the average squared differences between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: Square root of MSE, providing the error in the same unit as the target variable.
- **Accuracy**: R² score, which measures how well the model explains the variance in the target variable.

## Hyperparameter Tuning

For the **Random Forest** and **XGBoost** models, **GridSearchCV** was used to tune the hyperparameters to optimize performance. The parameter grids used for both models are:

- **RandomForest**:
  - `n_estimators`: Number of trees in the forest (50, 100, 200)
  - `max_depth`: Maximum depth of the trees (10, 20, None)
  - `min_samples_split`: Minimum samples required to split an internal node (2, 5)
  - `min_samples_leaf`: Minimum samples required to be at a leaf node (1, 2)

- **XGBoost**:
  - `learning_rate`: Step size at each iteration (0.01, 0.1, 0.2)
  - `n_estimators`: Number of boosting rounds (50, 100, 200)
  - `max_depth`: Maximum depth of a tree (3, 5, 7)

## Code Structure

### 1. **Data Preprocessing**
   - Load the dataset.
   - Label encode the `Date` column.
   - Split data into features (`X`) and target (`y`).
   - Perform **train-test split** (80-20 split).
   - Scale the features using **StandardScaler** for better model performance (especially for models like SVR).

### 2. **Model Training**
   - Define models: Linear Regression, Lasso, Ridge, Decision Tree, Random Forest, XGBoost, SVR.
   - Perform **GridSearchCV** for Random Forest and XGBoost to find the best hyperparameters.
   - Train each model using the scaled training data and evaluate it on the test data.

### 3. **Model Evaluation**
   - Calculate and store **MAE**, **MSE**, **RMSE**, and **R² (Accuracy)** for each model.
   - Print out the evaluation results to compare model performance.

### 4. **Model Selection**
   - Select the best model based on the **lowest MAE**.
   - Save the best model using **pickle** for future use.

## Results

| Model             | MAE  | MSE   | RMSE  | 
|-------------------|------|-------|-------|
| Linear Regression | 1.27 | 2.13  | 1.46  | 
| Lasso             | 1.27 | 2.13  | 1.46  | 
| Ridge             | 1.27 | 2.13  | 1.46  | 
| Decision Tree     | 1.72 | 4.40  | 2.10  | 
| Random Forest     | 1.28 | 2.12  | 1.46  |
| XGBoost           | 1.27 | 2.12  | 1.46  |
| SVR               | 1.29 | 2.18  | 1.48  |     

The best model based on MAE is **Lasso Regression**, with an MAE of 1.27.

## Saving the Best Model

The best-performing model (Lasso) is saved to a file using **pickle** for later use.

```python
import pickle
with open("best_traffic_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
```

## Future Improvements

- **Feature Engineering**: Adding more features such as traffic from previous days, holidays, and special events could improve model performance.
- **Model Tuning**: Further hyperparameter tuning for other models might improve performance.
- **Ensemble Learning**: Combining predictions from multiple models could lead to better accuracy.

## Conclusion

This project successfully built several machine learning models to predict traffic flow. The Lasso Regression model performed the best based on the evaluation metrics. This model can be integrated into an Intelligent Transportation System (ITS) to help optimize traffic management.

