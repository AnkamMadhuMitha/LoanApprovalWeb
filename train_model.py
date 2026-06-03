
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

df=pd.read_csv('loan_train.csv')
df=df.drop('Loan_ID',axis=1)

X=df.drop('Loan_Status',axis=1)
y=LabelEncoder().fit_transform(df['Loan_Status'])

cat=X.select_dtypes(include='object').columns
num=X.select_dtypes(exclude='object').columns

pre=ColumnTransformer([
('num',SimpleImputer(strategy='median'),num),
('cat',Pipeline([
('imp',SimpleImputer(strategy='most_frequent')),
('oh',OneHotEncoder(handle_unknown='ignore'))
]),cat)
])

model=Pipeline([
('pre',pre),
('rf',RandomForestClassifier(n_estimators=200,random_state=42))
])

Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
model.fit(Xtr,ytr)

joblib.dump(model,'loan_approval_model.pkl')
print("Model saved")
