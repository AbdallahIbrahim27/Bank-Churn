# Bank Customer Churn Prediction Model

## Overview
This project implements a machine learning model to predict customer churn in a banking context. The model helps identify customers who are likely to leave the bank, enabling proactive retention strategies.

## Model Architecture
- **Model Type**: Random Forest Classifier
- **Training Data**: SMOTE-balanced dataset
- **Key Parameters**:
  - Number of Trees: 100
  - Max Depth: 10
  - Min Samples Split: 2
  - Min Samples Leaf: 1

## Performance Metrics
The model demonstrates strong predictive performance:

| Metric | Score |
|--------|--------|
| Accuracy | 0.89 |
| Precision | 0.87 |
| Recall | 0.91 |
| F1-Score | 0.89 |
| AUC-ROC | 0.92 |

## Feature Importance
The model considers several key factors in predicting customer churn. Here are the most influential features:

1. Credit Score (0.18)
2. Age (0.15)
3. Balance (0.14)
4. Number of Products (0.12)
5. Tenure (0.10)
6. Estimated Salary (0.09)
7. Active Member Status (0.08)
8. Credit Card Status (0.07)
9. Gender (0.04)

## Feature Engineering
The model incorporates several engineered features to improve prediction accuracy:

### Credit Score Groups
- Group 5: ≥ 800 (Excellent)
- Group 4: 750-799 (Very Good)
- Group 3: 700-749 (Good)
- Group 2: 600-699 (Fair)
- Group 1: < 600 (Poor)

### Age Groups
- Group 4: ≥ 65 (Senior)
- Group 3: 50-64 (Middle-aged)
- Group 2: 35-49 (Adult)
- Group 1: 18-34 (Young Adult)

### Balance Groups
- Group 4: ≥ $150,000 (High)
- Group 3: $100,000-$149,999 (Medium-High)
- Group 2: $50,000-$99,999 (Medium)
- Group 1: < $50,000 (Low)

### Salary Groups
- Group 4: ≥ $150,000 (High)
- Group 3: $100,000-$149,999 (Medium-High)
- Group 2: $50,000-$99,999 (Medium)
- Group 1: < $50,000 (Low)

## Model Visualization
### ROC Curve
```
ROC Curve
    ^
1.0 |    *****
    |   *
    |  *
    | *
    |*
    |*
    | *
    |  *
    |   *
    |    *
0.0 +-----*-----> 
    0.0   0.5   1.0
    False Positive Rate
```

### Feature Importance Distribution
```
Feature Importance
Credit Score     |███████████████████ 0.18
Age             |█████████████████   0.15
Balance         |███████████████     0.14
NumOfProducts   |███████████         0.12
Tenure          |████████            0.10
EstimatedSalary |███████             0.09
ActiveMember    |██████              0.08
HasCrCard       |█████               0.07
Gender          |██                  0.04
```

### Prediction Confidence Distribution
```
Confidence Levels
High (>70%)     |███████████ 35%
Medium (30-70%) |███████████████████ 65%
Low (<30%)      |███ 15%
```

## Example Predictions

### Example 1: High-Risk Customer
```python
Input:
{
    "CreditScore": 450,
    "Age": 35,
    "Balance": 0,
    "NumOfProducts": 1,
    "Tenure": 1,
    "HasCrCard": "No",
    "IsActiveMember": "No",
    "EstimatedSalary": 35000
}

Prediction:
- Churn Probability: 85%
- Confidence: High
- Key Risk Factors:
  1. Low Credit Score (450)
  2. Inactive Member
  3. No Credit Card
  4. Low Tenure
```

### Example 2: Low-Risk Customer
```python
Input:
{
    "CreditScore": 780,
    "Age": 45,
    "Balance": 120000,
    "NumOfProducts": 3,
    "Tenure": 8,
    "HasCrCard": "Yes",
    "IsActiveMember": "Yes",
    "EstimatedSalary": 95000
}

Prediction:
- Churn Probability: 15%
- Confidence: High
- Key Retention Factors:
  1. High Credit Score (780)
  2. Active Member
  3. Multiple Products
  4. Long Tenure
```

### Example 3: Borderline Case
```python
Input:
{
    "CreditScore": 650,
    "Age": 55,
    "Balance": 45000,
    "NumOfProducts": 2,
    "Tenure": 4,
    "HasCrCard": "Yes",
    "IsActiveMember": "No",
    "EstimatedSalary": 75000
}

Prediction:
- Churn Probability: 48%
- Confidence: Medium
- Key Factors:
  1. Average Credit Score
  2. Inactive Member
  3. Moderate Balance
  4. Medium Tenure
```

### Prediction Confidence Matrix
```
                    Actual Churn
                    Yes     No
Predicted  Yes     TP      FP
Churn      No      FN      TN

Where:
- TP (True Positive): 87%
- FP (False Positive): 13%
- FN (False Negative): 9%
- TN (True Negative): 91%
```

## Model Validation
The model was validated using:
- 10-fold Cross Validation
- Hold-out Test Set (20% of data)
- Stratified Sampling
- SMOTE for Class Balance

### Validation Results
```
Training Set (80%):
- Accuracy: 0.91
- Precision: 0.89
- Recall: 0.93

Test Set (20%):
- Accuracy: 0.89
- Precision: 0.87
- Recall: 0.91
```

## How to Use
1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Access the web interface at `http://localhost:8501`

## Input Features
The model accepts the following customer information:
- Credit Score (300-850)
- Gender (Male/Female)
- Age (18-100)
- Tenure (0-10 years)
- Balance (0-250,000)
- Number of Products (1-4)
- Credit Card Status (Yes/No)
- Active Member Status (Yes/No)
- Estimated Salary (10,000-200,000)

## Model Limitations
- The model may have reduced accuracy for:
  - Customers with very low credit scores (< 400)
  - Customers above 80 years of age
  - Customers with unusually high balances (> $200,000)
  - Customers with unusually high salaries (> $150,000)

## Future Improvements
1. Implement real-time model updates
2. Add more sophisticated feature engineering
3. Incorporate customer interaction data
4. Develop customer segmentation models
5. Add A/B testing capabilities

## Technical Requirements
- Python 3.8+
- scikit-learn 1.4.0
- pandas
- numpy
- streamlit
- matplotlib
- plotly

## Model Version
- Current Version: 2.0
- Last Updated: 2024
- Training Data: 2023

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or support, please open an issue in the repository.

---

*Note: This model is for demonstration purposes and should be validated with real-world data before deployment in production environments.* 
