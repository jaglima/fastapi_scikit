from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from joblib import dump
import pathlib
import pandas as pd

# Reading raw data
df = pd.read_csv(pathlib.Path('data/dataset_test_ds_v2-Atualizado.csv'), encoding='latin-1')

# Data cleaning
df = df.dropna()
df.SAFRA = df.SAFRA.astype('str')
df.CEP = df.CEP.astype('str')

# Feature engineering
df['month'] = df.SAFRA.str[4:6]
df['month'] = df.month.astype(int)
df = pd.concat([df, pd.get_dummies(df.V11, prefix='UF')], axis=1)
df = pd.concat([df, pd.get_dummies(df.V12, prefix='city')], axis=1)
df = pd.concat([df, pd.get_dummies(df.CEP, prefix='zip')], axis=1)
df = pd.concat([df, pd.get_dummies(df.month, prefix='month')], axis=1)

## Train and test spliting
df = df.drop(['V11', 'V12', 'CEP', 'SAFRA', 'month'], axis=1)
df_train, df_test = train_test_split(df, test_size=0.30, stratify=df.TARGET, random_state=654321)

# Unbalanced dataset. Trying to overcome with oversampling the lower category
smote = SMOTE(random_state=654321)
X_train, y_train = smote.fit_resample(df_train, df_train.TARGET)
y_test = df_test.TARGET.astype('int')
X_train = X_train.drop(['TARGET'], axis=1)

# Feature selection ommited here. For more info, please, refer to the notebook attached
best_feats = ['V3',
             'V5',
             'V7',
             'V9',
             'UF_ES',
             'UF_MG',
             'UF_RJ',
             'UF_SP',
             'city_Belo Horizonte',
             'zip_26089250',
             'zip_29101685',
             'zip_8253410',
             'zip_8420400',
             'month_1',
             'month_2',
             'month_3',
             'month_4',
             'month_5',
             'month_6',
             'month_7',
             'month_8',
             'month_9',
             'month_10',
             'month_11',
             'month_12']

# Model training
model = LogisticRegressionCV(solver='liblinear', cv=10, scoring='f1', random_state=654321).fit(X_train[best_feats], y_train)

# Model serializing for API consuming
dump(model, pathlib.Path('model/model.joblib'))