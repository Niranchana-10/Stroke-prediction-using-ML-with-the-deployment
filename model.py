import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
df=pd.read_csv(r"C:\Users\DELL\Downloads\archive\healthcare-dataset-stroke-data.csv")
df.head()
df.drop(['id'], axis = 1, inplace = True)
df.info()
df.describe()
df['bmi'].fillna(df['bmi'].mean(),inplace=True)
df.describe()
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
gender=enc.fit_transform(df['gender'])
smoking_status=enc.fit_transform(df['smoking_status'])
work_type=enc.fit_transform(df['work_type'])
Residence_type=enc.fit_transform(df['Residence_type'])
ever_married=enc.fit_transform(df['ever_married'])
df['work_type']=work_type
df['ever_married']=ever_married
df['Residence_type']=Residence_type
df['smoking_status']=smoking_status
df['gender']=gender
y = df.stroke
X = df.drop('stroke', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, train_size =0.2)
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)
print("Accuracy_score",accuracy_score(y_test,y_pred))
print("Confusion matrix")
print(confusion_matrix(y_test,y_pred))
import pickle
with open('../stroke.pkl', 'wb') as file:
    pickle.dump(dt,file)

