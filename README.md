![Purple and Blue Gradient Ai Technology Banner (1)](https://github.com/user-attachments/assets/5865b9e4-79c8-46f0-b0ce-ee512209aebb)
# RetainAI: Leveraging ML to Boost Customer Retention

## Description
RetainAI is a data science project aimed at addressing customer churn in a telecom company. By performing exploratory data analysis (EDA), building a predictive model, and evaluating its performance, this project seeks to provide actionable insights that can help improve customer retention strategies.

## Project Flow

### Data Pre-processing & Exploratory Data Analysis
  
![customer_churn_prediction](https://github.com/user-attachments/assets/96bc5c1d-bb1d-4e35-aebb-53018c940b44)


### Feature Selection

**chi-square test**
```python
from scipy.stats import chi2_contingency
# chi-square test for area code and churn
crosstab_result = pd.crosstab(df['area code'], df['churn'])
chi2, p_value, dof, expec = chi2_contingency(crosstab_result)
print(p_value)
```
**One Way ANOVA**
```python
from scipy.stats import f_oneway

def one_way_anova_for_churn(feature):
    """
    Perform one-way ANOVA to test the relationship between a numerical feature and churn.
   
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    feature (str): The name of the numerical feature to test.
   
    Returns:
    str: 'Has Relation' if the p-value is <= 0.05, otherwise 'No Relation'.
    """
    # Create groups based on the 'churn' feature
    group_false = df[df['churn'] == False][feature]
    group_true = df[df['churn'] == True][feature]
   
    # Perform the one-way ANOVA
    f_stat, p_value = f_oneway(group_false, group_true)
   
    # Return results
    if p_value <= 0.05:
        return 'Has Relation'
    return 'No Relation'
```
**Correlation Analysis**
```python
df.corr()['churn']
```

### Model Building & Evaluation
```python
def train_eval_model(model):
   
    '''
    This function train & Evaluate the given model
   
    '''
    model.fit(X_train, y_train) # training
    y_pred_test = model.predict(X_test) #  test prediction
    y_pred_train = model.predict(X_train) #  training prediction
   
    print('\n',type(model).__name__)
   
    print('=='*25)
    print('\nTraining Accuracy :', accuracy_score(y_train, y_pred_train)) # training accuracy
    print('\nTest Accuracy :', accuracy_score(y_test, y_pred_test)) # test accuracy
    print('__'*25)
    print('\nConfusion Matrix:\n\n', confusion_matrix(y_test, y_pred_test)) # confusion matrix
    print('__'*25)
    print('\nClassification Report: \n')
   
    return  classification_report(y_test, y_pred_test) # classification report
```
## Contribution Guidelines
We welcome contributions from the community! To contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes and commit them (git commit -m 'Add new feature').
4. Push to the branch (git push origin feature-branch).
5. Open a pull request.

## License Information
This project does not have a specific license. Feel free to use the code as you see fit.

## Future Work
Potential future enhancements include:

1. Creating a robust pipeline for the model to streamline the process from data ingestion to prediction.
2. Deploying the model into production using cloud services.
3. Implementing additional machine learning algorithms for comparison.
4. Enhancing the EDA section with more visualizations and insights.

## Contact
For any questions or feedback, feel free to contact me at akashambure123@gmail.com  
Happy Coding! 🚀








