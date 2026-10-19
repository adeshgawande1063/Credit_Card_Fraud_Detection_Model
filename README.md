# Credit Card Fraud Detection Using Machine Learning

A machine learning project for identifying potentially fraudulent credit card transactions using a Random Forest classifier, with a focus on evaluation under severe class imbalance.

## Project Overview

Credit card fraud detection is an imbalanced classification problem: fraudulent transactions represent only a small fraction of all transactions. Accuracy alone can therefore be misleading.

This project explores transaction data, evaluates baseline and candidate classifiers, selects a model using Precision-Recall AUC (PR-AUC), performs hyperparameter tuning, and tests predictions on individual transactions.

## Objectives

- Explore and preprocess credit card transaction data.
- Compare multiple classification models using cross-validation.
- Select a model based on PR-AUC.
- Evaluate fraud detection using precision, recall, F1-score, and accuracy.
- Visualize performance using confusion matrices.
- Build a prediction function for individual transactions.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn

## Dataset

The project uses a credit card transaction dataset with 284,807 transactions and 30 columns, including the target label `Class`.

- `Class = 0`: Non-fraudulent transaction
- `Class = 1`: Fraudulent transaction

The dataset is not included in this repository because its CSV file exceeds GitHub's standard 100 MB file limit. Obtain the dataset from its authorized original source and place it in the project directory before running the notebook.

## Methodology

1. **Data preparation:** Load the dataset, separate features and target, and create training and test sets.
2. **Baseline evaluation:** Assess initial model performance using classification reports and average precision.
3. **Model comparison:** Compare Decision Tree, HistGradientBoosting, and Random Forest candidates using five-fold cross-validation with PR-AUC.
4. **Model selection:** Select Random Forest based on the model comparison results.
5. **Hyperparameter tuning:** Use `RandomizedSearchCV` with five-fold cross-validation and average precision as the scoring metric.
6. **Final evaluation:** Assess the tuned Random Forest on the held-out test set.
7. **Prediction:** Test the prediction function on individual transactions.

## Model Performance

### Final Random Forest — Test Set

| Metric | Result |
|---|---:|
| Accuracy | 99.95% |
| Fraud precision | 0.96 |
| Fraud recall | 0.76 |
| Fraud F1-score | 0.85 |
| Average Precision (PR-AUC) | 0.8679 |

The test set contains 56,962 transactions, including 98 fraudulent transactions.

### Cross-validation comparison

Mean PR-AUC across five folds, as reported in the notebook before hyperparameter tuning:

| Model | Mean CV PR-AUC |
|---|---:|
| Decision Tree (balanced) | 0.5366 |
| HistGradientBoosting (balanced) | 0.7462 |
| Random Forest (balanced) | 0.8359 |

Random Forest achieved the highest mean PR-AUC among these three candidates in the reported comparison.

*Note: Reported results are from the notebook's existing outputs. Results may vary if the dataset, split, or model configuration changes.*

## Confusion Matrix and Classification Report

The notebook includes confusion matrix visualizations and classification reports for the training and test sets. These help examine false positives and false negatives, which are particularly important in fraud detection.

## Example Predictions

The notebook's prediction function demonstrates two test transactions:

- A non-fraudulent transaction predicted as class `0`.
- A fraudulent transaction predicted as class `1`.

These examples illustrate the prediction workflow; they do not establish performance on every transaction.

## Repository Contents

- `Credit_Card_Fraud_Detection_Model.ipynb` — complete notebook containing preprocessing, model comparison, hyperparameter tuning, evaluation, visualizations, and predictions.
- `.gitignore` — excludes the large CSV dataset from Git tracking.

## How to Run

1. Clone or download this repository.
2. Obtain the dataset from its authorized original source.
3. Place the CSV file in the project directory using the filename expected by the notebook.
4. Install the libraries imported in the notebook.
5. Open `Credit_Card_Fraud_Detection_Model.ipynb` in Jupyter Notebook or VS Code.
6. Run the cells in order.

## Limitations

- The test set contains only 98 fraudulent transactions, so fraud-class metrics have some statistical uncertainty.
- High accuracy does not necessarily mean that fraud detection is effective; precision, recall, F1-score, and PR-AUC provide additional information.
- The reported training results are stronger than the test results. Further validation is useful to assess overfitting and generalization.
- Predictions should be evaluated in context and should not be treated as definitive proof of fraud.

## Future Improvements

- Evaluate recall and precision at different decision thresholds.
- Analyze false positives and false negatives.
- Add ROC curves and precision-recall curves.
- Validate performance across different data splits.
- Explore model explainability and reproducible deployment.

## Disclaimer

This project is intended for educational and research purposes. It is not a production-ready financial fraud detection system.

## Author

**Adesh Gawande**

GitHub: [adeshgawande1063](https://github.com/adeshgawande1063)
