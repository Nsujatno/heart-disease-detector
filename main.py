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

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# random int for our hyper parameter tuning
from scipy.stats import randint, loguniform

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
# prints first row
print(data.iloc[0])

# create splits
X = data.drop(columns=["num"])
y = data["num"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# train the random forest model
model_rf = RandomForestClassifier(random_state=42)
param_dist_rf = {
    'n_estimators': randint(100, 1000), # num of decision trees in the forest
    'max_depth': [None, 10, 20, 30, 40, 50], # num of "questions" each tree is allowed to have, none means that it can grow as deep as it wants
    'min_samples_split': randint(2, 15),
    'min_samples_leaf': randint(1, 10), # min num of data points allowed to be in final leaf
    'max_features': ['sqrt', 'log2'], # the num of features to consider when looking for best split
    'bootstrap': [True, False] # Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
}
search = RandomizedSearchCV(estimator=model_rf,
                            param_distributions=param_dist_rf,
                            n_iter=100,
                            scoring='accuracy',
                            n_jobs=-1,
                            random_state=42)
model_rf.fit(X_train, y_train)
search.fit(X_train, y_train)
# gets the best model and the best parameters
best_model = search.best_estimator_
best_params_ = search.best_params_

# train the gradient boost model
model_gbm = GradientBoostingClassifier(random_state=42)
param_gbm = {
    'n_estimators': randint(100, 1000),
    'learning_rate': loguniform(0.005, 0.3),
    'max_depth': randint(3,10)
}
search_gbm = RandomizedSearchCV(estimator=model_gbm,
                                param_distributions=param_gbm,
                                n_iter=50,
                                n_jobs=-1,
                                random_state=42)
model_gbm.fit(X_train, y_train)
search_gbm.fit(X_train, y_train)
best_model_gbm = search_gbm.best_estimator_


# evaluate
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

prediction_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, prediction_rf)

prediction_best_rf = best_model.predict(X_test)
accuracy_best_rf = accuracy_score(y_test, prediction_best_rf)

prediction_gbm = model_gbm.predict(X_test)
accuracy_gbm = accuracy_score(y_test, prediction_gbm)

prediction_best_gbm = best_model_gbm.predict(X_test)
accuracy_best_gbm = accuracy_score(y_test, prediction_best_gbm)

print(f'Accuracy of logistic regression: {accuracy}')
print(f'Accuracy of random forest classifier {accuracy_rf}')
print(f'Accuracy of best random forest classifier using randomized search cv and hyperparameter tuning {accuracy_best_rf}')
print(f'Accuracy of gbm: {accuracy_gbm}')
print(f'Accuracy of best gbm {accuracy_best_gbm}')