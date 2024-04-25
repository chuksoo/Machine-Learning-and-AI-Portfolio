import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier 
from sklearn.metrics import roc_auc_score, log_loss, f1_score

# set random seed
seed = 42

# read in the data
transactions = pd.read_json('https://github.com/CapitalOneRecruiting/DS/blob/master/transactions.zip?raw=true', compression='zip', lines=True)

# data preprocessing steps
def change_datatype(df, cols, type_val):
    for col in cols:
        df[col] = df[col].astype(type_val)
        
change_datatype(transactions, ['transactionDateTime', 'currentExpDate', 'accountOpenDate', 'dateOfLastAddressChange'], 'datetime64[ns]')
change_datatype(transactions, ['accountNumber', 'customerId', 'creditLimit', 'cardCVV', 'enteredCVV', 'cardLast4Digits'], 'int32')
change_datatype(transactions, ['availableMoney', 'transactionAmount', 'currentBalance'], 'float32')

def create_features(data):
    data['transactionMonth'] = data['transactionDateTime'].dt.month_name()
    data['transactionDayofWeek'] = data['transactionDateTime'].dt.day_name()
    data['transactionHour'] = data['transactionDateTime'].dt.hour
    data['transactionMinutes'] = data['transactionDateTime'].dt.minute
    data['transactionSeconds'] = data['transactionDateTime'].dt.second
    data['currentExpMonth'] = data['currentExpDate'].dt.month_name()
    data['currentExpDayofWeek'] = data['currentExpDate'].dt.day_name()
    data['accountOpenMonth'] = data['accountOpenDate'].dt.month_name()
    data['accountOpenDayofWeek'] = data['accountOpenDate'].dt.day_name()
    data['dateOfLastAddressChangeMonth'] = data['dateOfLastAddressChange'].dt.month_name()
    data['dateOfLastAddressChangeDayofWeek'] = data['dateOfLastAddressChange'].dt.day_name()

# create new feature
create_features(transactions)
change_datatype(transactions, ['transactionHour', 'transactionMinutes', 'transactionSeconds'], 'int32') 

# identify reversed transaction
transactions['isDuplicated'] = (transactions.sort_values(['transactionDateTime'])
                                .groupby(['customerId', 'transactionAmount'], sort=False)['transactionDateTime']
                                .diff()
                                .dt.total_seconds()
                                .lt(120)
                               )

# drop duplicated transactions from the data
transaction_duplicate = transactions.query('isDuplicated == True')
rows = transaction_duplicate.index
card_transactions = transactions.drop(index = rows)

# Get sample of the data without replacement 
transactions_df = card_transactions.sample(frac=0.5, replace=False, random_state=42)

# drop unimportant features
transactions_df = transactions_df.drop(['transactionDateTime', 'currentExpDate', 'accountOpenDate', 'dateOfLastAddressChange', 'cardCVV', 'enteredCVV', 
                                        'cardLast4Digits', 'echoBuffer', 'merchantCity', 'merchantState', 'merchantZip', 'posOnPremises', 'recurringAuthInd', 'isDuplicated'], axis=1)

# encode some features
transactions_df['expirationDateKeyInMatch'] = transactions_df['expirationDateKeyInMatch'].replace({True: 1, False: 0})
transactions_df['cardPresent'] = transactions_df['cardPresent'].replace({True: 1, False: 0})
transactions_df['isFraud'] = transactions_df['isFraud'].replace({True: 1, False: 0})

# create features and target
features = transactions_df.drop(['isFraud'], axis = 1)
target = transactions_df.isFraud

# Get numerical and categorical features
cat_feature_cols = [cname for cname in features.columns if features[cname].dtype == "object"]
num_feature_cols = [cname for cname in features.columns if features[cname].dtype in ["int64", "int32", "float64", "float32"]] 

# preprocessing pipelines
num_pipeline = Pipeline(
    steps=[("scaler", StandardScaler())
    ]
)

cat_pipeline = Pipeline(
    steps=[('encoding', OrdinalEncoder())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num_pipeline", num_pipeline, num_feature_cols),
        ("cat_pipeline", cat_pipeline, cat_feature_cols)
        ]
    )

# Apply all the stages of transformation to the data
preprocessed_data = preprocessor.fit_transform(features)

# Get feature names after transformation
preprocessed_features = pd.DataFrame(preprocessed_data, columns=num_feature_cols + cat_feature_cols)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_features, target, test_size=0.20, random_state=12345)

# build the model
lgbm_model = LGBMClassifier(learning_rate=0.1, n_estimators=200, num_leaves=31, verbose=0)

# train the model on training data
lgbm_model.fit(X_train, y_train) 

# predict on test data
lgbm_pred = lgbm_model.predict(X_test)
lgbm_pred_proba = lgbm_model.predict_proba(X_test)[:, 1]

# calculate metrics
print('ROC-AUC Score: {:.3f}'.format(roc_auc_score(y_test, lgbm_pred_proba, multi_class='ovr')))
print('Log loss is: {:.3f}'.format(log_loss(y_test, lgbm_pred_proba)))
print('F1 score: {:.3f}'.format(f1_score(y_test, lgbm_pred, average="weighted")))

# save the model to disk
with open('./streamlit_app/transactions-deploy/transactions_lgbm_model.sav', 'wb') as model_file:
    joblib.dump(lgbm_model, model_file)
