# installations
# pip install pandas numpy matplotlib seaborn scikit-learn

# data info
# id (Unique id for each patient)
# age (Age of the patient in years)
# origin (place of study)
# sex (Male/Female)
# cp chest pain type ([typical angina, atypical angina, non-anginal, asymptomatic])
# trestbps resting blood pressure (resting blood pressure (in mm Hg on admission to the hospital))
# chol (serum cholesterol in mg/dl)
# fbs (if fasting blood sugar > 120 mg/dl)
# restecg (resting electrocardiographic results)
# -- Values: [normal, stt abnormality, lv hypertrophy]
# thalach: maximum heart rate achieved
# exang: exercise-induced angina (True/ False)
# oldpeak: ST depression induced by exercise relative to rest
# slope: the slope of the peak exercise ST segment
# ca: number of major vessels (0-3) colored by fluoroscopy
# thal: [normal; fixed defect; reversible defect]
# num: the predicted attribute

# imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# create dataframe
data = pd.read_csv("heart_disease_uci.csv")
# print(data.info())

# print any missing values
# print(data.isnull().sum())

# statistical analysis of the data
# print(data.describe())

def preprocess_data(df):
    # target variable
    df["num"] = np.where(df['num'] > 0, 1, 0)
    df.drop(columns=["id", "dataset"], inplace=True)
    numerical_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

    # replace missing data in numerical with the median of the column
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # replace missing data in categorical with most frequent
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # map male, female to 1 and 0
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
    # map true and false to 1 and 0
    df["fbs"] = df["fbs"].map({True: 1, False: 0})
    df["exang"] = df["exang"].map({True: 1, False: 0})

    # one hot encoding
    encoding_cols = ["cp", "restecg", "slope", "thal"]
    df = pd.get_dummies(df, columns=encoding_cols, drop_first=True)

    return df

data = preprocess_data(data)
print(data.head())
print(data.iloc[0])