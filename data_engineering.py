import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

'''
Feature Engineering File
'''

def create_application_features(df):
    df = df.copy()

    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25

    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"] /
        df["AMT_INCOME_TOTAL"].replace(0, np.nan))

    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"] /
        df["AMT_INCOME_TOTAL"].replace(0, np.nan))

    df["CREDIT_ANNUITY_RATIO"] = (
        df["AMT_CREDIT"] /
        df["AMT_ANNUITY"].replace(0, np.nan))

    df["GOODS_CREDIT_RATIO"] = (
        df["AMT_GOODS_PRICE"] /
        df["AMT_CREDIT"].replace(0, np.nan))

    df["INCOME_PER_PERSON"] = (
        df["AMT_INCOME_TOTAL"] /
        df["CNT_FAM_MEMBERS"].replace(0, np.nan))

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


'''
Simple features from historical credit and repayment data.
'''

def create_historical_features(bureau, previous_application, installments):

    #Bureau: Simple aggregation features
    bureau_features = bureau.groupby("SK_ID_CURR").agg(BUREAU_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
                                                       BUREAU_TOTAL_CREDIT=("AMT_CREDIT_SUM", "sum"),
                                                       BUREAU_AVG_CREDIT=("AMT_CREDIT_SUM", "mean"),
                                                       BUREAU_TOTAL_DEBT=("AMT_CREDIT_SUM_DEBT", "sum"),
                                                       BUREAU_MAX_OVERDUE=("AMT_CREDIT_MAX_OVERDUE", "max"),
                                                       BUREAU_AVG_DAYS_CREDIT=("DAYS_CREDIT", "mean")).reset_index()

    #Previous applications: Simple aggregation features
    previous_features = previous_application.groupby("SK_ID_CURR").agg(PREVIOUS_APPLICATION_COUNT=("SK_ID_PREV", "count"),
                                                                       PREVIOUS_AVG_CREDIT=("AMT_CREDIT", "mean"),
                                                                       PREVIOUS_MAX_CREDIT=("AMT_CREDIT", "max"),
                                                                       PREVIOUS_AVG_ANNUITY=("AMT_ANNUITY", "mean"),
                                                                       PREVIOUS_AVG_GOODS_PRICE=("AMT_GOODS_PRICE", "mean")).reset_index()


    #Previous intallements: Simple aggregation features
    installment_features = installments.groupby("SK_ID_CURR").agg(INSTALLMENT_RECORD_COUNT=("SK_ID_PREV", "count"),
                                                                  AVG_INSTALLMENT=("AMT_INSTALMENT", "mean"),
                                                                  MAX_INSTALLMENT=("AMT_INSTALMENT", "max"),
                                                                  AVG_ACTUAL_PAYMENT=("AMT_PAYMENT", "mean"),
                                                                  MAX_ACTUAL_PAYMENT=("AMT_PAYMENT", "max")).reset_index()

    return (bureau_features,previous_features,installment_features)



'''
Create an updated features dataset
'''
def build_dataset(application_train,bureau,previous_application,installments,missing_threshold=0.50):

    df = create_application_features(application_train)

    (bureau_features,previous_features,installment_features) = create_historical_features(bureau,previous_application,installments)

    # Join the datasaets on the main ID key
    df = df.merge(bureau_features, on="SK_ID_CURR", how="left")
    df = df.merge(previous_features, on="SK_ID_CURR", how="left")
    df = df.merge(installment_features, on="SK_ID_CURR", how="left")

    #Consitstency check
    assert len(df) == len(application_train)

    # Remove constant columns
    constant_cols = df.columns[df.nunique(dropna=False) <= 1].tolist()
    constant_cols = [c for c in constant_cols if c not in ["TARGET", "SK_ID_CURR"]]

    # Remove columns where missing features are above threshold value
    high_missing_cols = df.columns[df.isna().mean() > missing_threshold].tolist()
    high_missing_cols = [c for c in high_missing_cols if c not in ["TARGET", "SK_ID_CURR"]]

    # Drop filtered and redundant features
    df = df.drop(columns=constant_cols + high_missing_cols + ["DAYS_BIRTH", "DAYS_EMPLOYED"], errors="ignore")

    return df

'''
Prepare the data for models
'''
def prepare_train_test(df, test_size=0.20, random_state=42):
    y = df["TARGET"].astype(int)
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, df["SK_ID_CURR"], test_size=test_size, stratify=y, random_state=random_state)
    X_train, X_test = X_train.copy(), X_test.copy()

    # Split features by their type
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

    # Create Missing Value Indicator features in order to capture the significance for the model.
    missing_numeric_cols = [col for col in numeric_cols if X_train[col].isna().any()]
    for col in missing_numeric_cols:
        X_train[f"{col}_MISSING"] = X_train[col].isna().astype(int)
        X_test[f"{col}_MISSING"] = X_test[col].isna().astype(int)

    # Impute the missing values for all columns, too simple but just to get initial prototype working.
    train_medians = X_train[numeric_cols].median()
    for data in (X_train, X_test):
        data[numeric_cols] = data[numeric_cols].fillna(train_medians)
        data[categorical_cols] = data[categorical_cols].fillna("Missing")

    # Ensure data is free of .nan values
    assert X_train.isna().sum().sum() == 0 and X_test.isna().sum().sum() == 0

    # Logging the new df information
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train default rate: {y_train.mean():.4f} | Test default rate: {y_test.mean():.4f}")
    print(f"Missing indicators: {len(missing_numeric_cols)}")

    return X_train, X_test, y_train, y_test, id_train, id_test
