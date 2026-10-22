# Credit Risk Modelling

A small end-to-end credit risk modelling project using the Home Credit Default Risk dataset.

The project compares three modelling approaches using the same engineered features and Train/Test population:

- Logistic Regression
- LightGBM
- Neural Network

The focus is not only predictive performance, but also data preparation, generalisation, interpretability and model judgement.

## Live Demo

[Open the Streamlit Project](https://credit-risk---default-project-tra3rxzz3srlwlljcvb7zt.streamlit.app/)

## Data

The project uses the Home Credit Default Risk dataset.

The raw data is relational and includes the current application together with historical credit, previous application and repayment information.

The original data files are not included in this repository. They should be downloaded separately and stored under `data/raw/`.

## Data Preparation

The common data-engineering pipeline is shared across all three models.

The main steps are:

- inspect missingness, duplicates and feature types;
- treat the anomalous values;
- create simple application-level features such as age, employment duration and affordability ratios;
- aggregate bureau, previous-application and instalment history to applicant level;
- remove constant and highly incomplete features;
- create numerical missingness indicators;
- learn numerical imputation values from the Train population only;
- create a stratified Train/Test split.

Feature engineering is deliberately kept relatively small and interpretable rather than optimised for Kaggle leaderboard performance.

## Models

### Logistic Regression

Used as the transparent statistical benchmark.

Model-specific preprocessing includes standardisation of numerical variables and one-hot encoding of categorical variables. The fitted coefficients and odds ratios are inspected to understand the direction and relative strength of model relationships.

### LightGBM

Used as the main nonlinear tree-based challenger.

Numerical scaling is not required. Feature importance and SHAP are used to investigate which variables the model relies on and how features contribute to predictions.

### Neural Network

Used as a second nonlinear challenger.

Numerical variables are standardised and categorical variables are one-hot encoded. The network contains two hidden layers with dropout and is mainly used as a performance challenger because it is less directly interpretable than the other approaches.

## Model Evaluation

All models are compared on the same Test population using:

- ROC-AUC
- KS statistic
- Brier score
- Train and Test performance

The dataset is imbalanced, with Default applicants representing roughly 8% of the population. Class weighting and oversampling were explored as sensitivity checks rather than assumed to improve the models automatically.

## Results

The main results from the current experiment are:

| Model | Train AUC | Test AUC | Test KS | Test Brier |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7531 | 0.7518 | 0.3745 | 0.0682 |
| LightGBM | 0.8518 | 0.7732 | 0.4095 | 0.0666 |
| Neural Network | 0.7761 | 0.7586 | 0.3834 | 0.0677 |

LightGBM provides the strongest Test AUC and KS and the lowest Brier score. Its Train performance is noticeably higher than its Test performance, suggesting some overfitting.

Logistic Regression provides the most stable performance and remains the clearest model to interpret directly.

The Neural Network improves on Logistic Regression but does not outperform LightGBM in the current experiment.

## Explainability

Interpretability is treated differently across the models:

- **Logistic Regression:** coefficients and odds ratios provide direct feature-level interpretation.
- **LightGBM:** feature importance shows which variables the model relies on, while SHAP gives additional information about the direction and size of feature contributions.
- **Neural Network:** used primarily as a challenger because the internal network weights are not directly interpretable in the same way.

External credit scores appear as important risk-reducing features in both Logistic Regression and LightGBM, providing useful consistency across the two model families.

## Streamlit Application

The Streamlit application provides an interactive summary of the project with five sections:

1. Project Overview
2. Model Comparison
3. Explainability
4. Conclusion

The application reads saved model outputs rather than retraining the full modelling pipeline whenever the application is opened.

Run the application locally with:

```bash
streamlit run app.py
```

## Installation

Create and activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

The modelling notebooks and application were developed using Python libraries including pandas, NumPy, scikit-learn, LightGBM, PyTorch, SHAP, Matplotlib, Seaborn and Streamlit.

## Limitations

This is a modelling prototype built using public data rather than a production lending system.

Important limitations include:

- the  target is not treated as a formal regulatory Definition of Default;
- the validation is based on a random Train/Test split rather than a genuine out-of-time sample;
- feature engineering and hyperparameter tuning are intentionally limited;
- a full calibration or recalibration exercise has not been carried out;
- the project does not include a genuine post-deployment monitoring population;
- further fairness, stability, sensitivity and governance analysis would be required before real lending use.

## Further Work

Natural extensions include:

- cross-validation;
- stronger LightGBM regularisation;
- out-of-time validation;
- calibration and recalibration;
- segment-level validation;
- stability and drift monitoring;
- additional feature governance;
- fuller model documentation and independent validation.

## Purpose

This project was built as a compact demonstration of the credit-risk modelling lifecycle: taking raw relational data, constructing a modelling population, comparing statistical and machine-learning approaches, checking results, interpreting model behaviour and presenting the conclusions in a reusable application.
