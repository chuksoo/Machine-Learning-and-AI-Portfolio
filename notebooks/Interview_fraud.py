#!/usr/bin/env python
# coding: utf-8

# ### Make a prediction for the amount and predict the fraud use 

# In[1]:


import pandas as pd
import numpy as np

# matplotlib for visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# seaborn for statistical data visualization
import seaborn as sns

# import machine learning models
from sklearn.linear_model import LinearRegression # import linear regression algorithm
from sklearn.ensemble import RandomForestRegressor # import random forest algorithm
from catboost import CatBoostRegressor, Pool # import catboost regressor
from lightgbm import LGBMRegressor # import lightgbm regressor
from xgboost import XGBRegressor # import xgboost regressor
from sklearn.neighbors import KNeighborsRegressor # import kneighbor regression

# import metric to measure quality of model
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# modules for data preprocessing and pipelines 
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTE, SMOTENC
pd.options.mode.chained_assignment = None # to avoid SettingWithCopyWarning after scaling

# remove warnings
import warnings
warnings.filterwarnings('ignore')

print('Project libraries has been successfully been imported!')


# In[2]:


# read the dataset
interview_fraud_df = pd.read_excel('C:/Users/hotty/Documents/Personal Projects/Machine Learning Projects/Machine-Learning-and-AI-Portfolio/data/raw/interview_fraud.xlsx').drop(['Unnamed: 0'],axis=1)
interview_fraud_df.head()


# ### Data Exploration

# In[3]:


# describe the data
interview_fraud_df.describe()


# In[4]:


# check information about the data
interview_fraud_df.info()


# In[5]:


# find count of unique payment type
unique_type_percent = (interview_fraud_df['type'].value_counts() / interview_fraud_df['type'].value_counts().sum() * 100).tolist()   

# unique type
unique_type = interview_fraud_df['type'].value_counts().reset_index().rename(columns={'index': 'type', 'type': 'unique count'})
unique_type['percentage split (%)'] = ['{:.2f}'.format(x) for x in unique_type_percent]
unique_type


# In[6]:


# group data by type
group_amount = interview_fraud_df.groupby('type').agg({'amount': 'sum'}).sort_values(by = 'amount', ascending = False).reset_index()
group_amount


# In[7]:


# check value count of isFlaggedFraud and isFraud
interview_fraud_df['isFlaggedFraud'].value_counts()


# In[8]:


# check value count of isFraud
interview_fraud_df['isFraud'].value_counts()


# We see that the `isFlaggedFraud` feature contains 100000 zeros, so we need to drop that feature. Let's check correlation within features of the data

# In[9]:


# Plot histogram of parameters
interview_fraud_df[['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
       'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud']].hist(bins=30, figsize=(15, 10))
plt.suptitle('Histogram of selected parameters', y=0.95);


# In[10]:


# correlation matrix of features
plt.figure(figsize=(8, 6))
corrMatrix = interview_fraud_df.corr()
sns.heatmap(corrMatrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Plot for certain features')
plt.show();


# We can see that there is high correlation between `oldbalanceDest` and `newbalanceDest` and also between `oldbalanceOrig` and `newbalanceOrig`.

# ### Modeling Process

# In[11]:


# Extract the features and target variable
fraud_df = interview_fraud_df.copy()

# drop unimportant features
fraud_df = fraud_df.drop(['isFlaggedFraud'], axis=1)
fraud_df

# create features and target for predicting amount
amount_features = fraud_df.drop(['amount'], axis=1)
amount_target = fraud_df.amount


# In[12]:


amount_features


# In[13]:


amount_features.info()


# In[14]:


# split categorical columns based on cardinality
def split_categorical_features(df, n=10):
    """
    Splits categorical columns into 2 lists based on cardinality (i.e # of unique values)
    """
    high_cardinality_cols = [cname for cname in df.columns if df[cname].nunique() > n and df[cname].dtype == 'object']
    low_cardinality_cols = [cname for cname in df.columns if df[cname].nunique() < n and df[cname].dtype == 'object']
    return low_cardinality_cols, high_cardinality_cols

# function to preprocess data
def data_preprocessing_pipeline(df):
    # numerical and categorical features
    categorical_cols = [cname for cname in df.columns if df[cname].dtype == 'object']  # categorical columns 
    numeric_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'int32', 'float64', 'float32']]  # numerical columns

    # convert categorical features to string 
    df[categorical_cols] = df[categorical_cols].astype('object')
    
    # train set features split by cardinality
    categorical_low, categorical_high = split_categorical_features(df)
    
    # changing the feature to a category type for ordinal encoding
    for i in categorical_high:
        df[i] = df[i].astype('category')
        df[i] = df[i].cat.codes
    
    # preprocessing pipelines
    # continuos pipeline
    numeric_transformer = Pipeline(
        steps=[('scaler', StandardScaler())]
    )

    # categorical low pipeline
    categorical_transformer_low = Pipeline(
        steps=[
            ('encoding', OrdinalEncoder())
        ]
    )

    # categorical high pipeline
    categorical_transformer_high = Pipeline(
        steps=[
            ('encoding', OrdinalEncoder()),
            ('scaler', StandardScaler())
        ]
    )
    
    # create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_cols),
            ("categorical_low", categorical_transformer_low, categorical_low),
            ("categorical_high", categorical_transformer_high, categorical_high),
        ]
    )
    
    return preprocessor


# In[15]:


# function to extract feature names from pipeline
def get_feature_out(estimator, feature_in):
    if hasattr(estimator,'get_feature_names'):
        if isinstance(estimator, _VectorizerMixin):
            # handling all vectorizers
            return [f'vec_{f}'                 for f in estimator.get_feature_names_out()] 
        else:
            return estimator.get_feature_names_out(feature_in)  
    elif isinstance(estimator, SelectorMixin):
        return np.array(feature_in)[estimator.get_support()]
    else:
        return feature_in


def get_ct_feature_names(ct):
    # handles all estimators, pipelines inside ColumnTransfomer
    # doesn't work when remainder =='passthrough'
    # which requires the input column names.
    output_features = []

    for name, estimator, features in ct.transformers_:
        if name!='remainder':
            if isinstance(estimator, Pipeline):
                current_features = features
                for step in estimator:
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(estimator, features)
            output_features.extend(features_out)
        elif estimator=='passthrough':
            output_features.extend(ct._feature_names_in[features])
                
    return output_features


# In[16]:


# preprocess dataset for modeling
preprocessor = data_preprocessing_pipeline(amount_features)
preprocessor


# In[17]:


# final preprocessed data pipeline
preprocessed_amount_df = preprocessor.fit_transform(amount_features)
preprocessed_amount = pd.DataFrame(preprocessed_amount_df, columns=get_ct_feature_names(preprocessor))
preprocessed_amount.head()


# ### Split data into 60% training, 20% validation and 20% testing sets
# Here we split the data into training, validation and testing sets in the ratio 60:20:20 respectively.

# In[18]:


# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_amount, amount_target, test_size=0.20, random_state=12345)

# split train data into validation and train 
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.25, random_state = 12345) # 0.25 * 0.80 = 0.20 for validation size

# display the shape of the split dataset
print('The train set now contains {}'.format(X_train.shape[0]) + ' dataset representing 60% of the data') 
print('The valid set now contains {}'.format(X_valid.shape[0]) + ' dataset representing 20% of the data')
print('The test set now contains {}'.format(X_test.shape[0]) + ' dataset representing 20% of the data')


# ### Train Linear Regressor

# In[19]:


# function to train model and make predictions
def train_linear_model(X_train, y_train):
    """This function trains a linear model"""
    global lr_model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
def linear_regressor_prediction(X_test, y_test):
    """
    This function is used to make prediction 
    using a linear regression model
    """
    lr_pred = lr_model.predict(X_test)
    # rmse for linear model
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    print("\033[1m" + 'RMSE using Linear Regression' + "\033[0m")
    print('RMSE: {:.3f}'.format(lr_rmse))
    print()


# In[20]:


get_ipython().run_cell_magic('time', '', '# train linear model\ntrain_linear_model(X_train, y_train)')


# In[21]:


get_ipython().run_cell_magic('time', '', '# make predictions with linear regression for validation data\nlinear_regressor_prediction(X_valid, y_valid)')


# ### Train KNeighbors Regressor

# In[22]:


get_ipython().run_cell_magic('time', '', '# hyperparameter optimization for KNeighbors regression\n\n# define hyperparameters to tune\nknn_grid = {\'n_neighbors\' : range(1,5,1),\n            \'algorithm\' : [\'auto\', \'ball_tree\', \'kd_tree\', \'brute\'],\n            }\n# define the model \nknn_regr = KNeighborsRegressor()\n\n# define the grid search\ngrid_search_knn = GridSearchCV(\n    estimator = knn_regr, \n    param_grid = knn_grid, \n    scoring = "neg_mean_squared_error", \n    cv = 5, \n    n_jobs = 1\n)\n# execute search\ngrid_search_knn.fit(X_train, y_train)\n# summarize result\nprint(\'The best hyperparameters are: {}\'.format(grid_search_knn.best_params_))')


# In[23]:


# function to train model and make predictions
def train_KNeighbors_regressor(X_train, y_train):
    """This function trains a KNeighbors regressor model"""
    global knn_model
    # build the model
    knn_model = KNeighborsRegressor(**grid_search_knn.best_params_)
    knn_model.fit(X_train, y_train) # train the model 
    
def KNeighbors_regressor_prediction(X_test, y_test):
    """
    This function is used to make prediction 
    using the KNeighbors regression model
    """
    knn_pred = knn_model.predict(X_test)
    # determine RMSE for KNeighbors regressor
    knn_rmse = np.sqrt(mean_squared_error(y_test, knn_pred))
    print("\033[1m" + 'RMSE using KNeighbors Regressor' + "\033[0m")
    print('RMSE: {:.3f}'.format(knn_rmse))
    print()


# In[24]:


get_ipython().run_cell_magic('time', '', '# train KNeighbors regressor model\ntrain_KNeighbors_regressor(X_train, y_train)')


# In[25]:


get_ipython().run_cell_magic('time', '', '# make predictions with KNeighbors regressor for validation data\nKNeighbors_regressor_prediction(X_valid, y_valid)')


# ### Train LightGBM Regressor

# In[26]:


get_ipython().run_cell_magic('time', '', '# hyperparameter optimization for LightGBM regression\n\n# define hyperparameters to tune\nlgbm_grid = {\'learning_rate\': [0.001, 0.01, 0.05, 0.1],\n             \'n_estimators\': [50, 100, 500],\n             \'num_leaves\': [5, 10, 20, 31]\n            }\n# define the model \nlgbm_regr = LGBMRegressor(random_state = 12345)\n\n# define the grid search\ngrid_search_lgbm = GridSearchCV(\n    estimator = lgbm_regr, \n    param_grid = lgbm_grid, \n    scoring = "neg_mean_squared_error", \n    cv = 5, \n    n_jobs = 1\n)\n# execute search\ngrid_search_lgbm.fit(X_train, y_train)\n# summarize result\nprint(\'The best hyperparameters are: {}\'.format(grid_search_lgbm.best_params_))')


# In[27]:


# function to train model and make predictions
def train_lightGBM_regressor(X_train, y_train):
    """This function trains a LightGBM regressor model"""
    global lgbm_model
    # build the model
    lgbm_model = LGBMRegressor(**grid_search_lgbm.best_params_)
    lgbm_model.fit(X_train, y_train) # train the model 
    
def lightGBM_regressor_prediction(X_test, y_test):
    """
    This function is used to make prediction 
    using the lightGBM regression model
    """
    lgbm_pred = lgbm_model.predict(X_test)
    # determine RMSE for LightGBM regressor
    lgbm_rmse = np.sqrt(mean_squared_error(y_test, lgbm_pred))
    print("\033[1m" + 'RMSE using LightGBM Regressor' + "\033[0m")
    print('RMSE: {:.3f}'.format(lgbm_rmse))
    print()
    # feature importance from LightGBM regression 
    sorted_feature_importance = lgbm_model.feature_importances_.argsort()
    plt.figure(figsize=(8,6))
    plt.barh(X_train.columns[sorted_feature_importance], 
             lgbm_model.feature_importances_[sorted_feature_importance], 
             color='turquoise')
    plt.xlabel("LightGBM Feature Importance")


# In[28]:


get_ipython().run_cell_magic('time', '', '# train lightGBM regressor model\ntrain_lightGBM_regressor(X_train, y_train)')


# In[29]:


get_ipython().run_cell_magic('time', '', '# make predictions with lightGBM regressor for validation data\nlightGBM_regressor_prediction(X_valid, y_valid)')


# ### Train XGBoost Regressor

# In[30]:


get_ipython().run_cell_magic('time', '', '# hyperparameter optimization for XGBoost\n\n# define hyperparameters to tune\nxgb_grid = {\'learning_rate\': [0.001, 0.1, 0.3], \n            \'max_depth\': [4, 6, 10]\n           }\n# define the model \nxgb_regr = XGBRegressor(random_state = 12345)\n\n# define the grid search\ngrid_search_xgb = GridSearchCV(\n    estimator = xgb_regr, \n    param_grid = xgb_grid, \n    scoring = "neg_mean_squared_error", \n    cv = 5, \n    n_jobs = 1\n)\n# execute search\ngrid_search_xgb.fit(X_train, y_train)\n# summarize result\nprint(\'The best hyperparameters are: {}\'.format(grid_search_xgb.best_params_))')


# In[31]:


# function to train model and make predictions
def train_xgboost_regressor(X_train, y_train):
    """This function trains a XGBoost regressor model"""
    global xgb_model
    # build the model
    xgb_model = XGBRegressor(**grid_search_xgb.best_params_)
    xgb_model.fit(X_train, y_train) # train the model 
    
def xgboost_regressor_prediction(X_test, y_test):
    """
    This function is used to make prediction 
    using the XGBoost regression model
    """
    xgb_pred = xgb_model.predict(X_test)
    # determine RMSE for XGBoost regressor
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    print("\033[1m" + 'RMSE using XGBoost Regressor' + "\033[0m")
    print('RMSE: {:.3f}'.format(xgb_rmse))
    print()
    # feature importance from XGBoost regression 
    sorted_feature_importance = xgb_model.feature_importances_.argsort()
    plt.figure(figsize=(8,6))
    plt.barh(X_train.columns[sorted_feature_importance], 
             xgb_model.feature_importances_[sorted_feature_importance], 
             color='turquoise')
    plt.xlabel("XGBoost Feature Importance")


# In[32]:


get_ipython().run_cell_magic('time', '', '# train xgboost regressor model\ntrain_xgboost_regressor(X_train, y_train)')


# In[33]:


get_ipython().run_cell_magic('time', '', '# make predictions with xgboost regressor for validation data\nxgboost_regressor_prediction(X_valid, y_valid)')


# In[ ]:





# ### Train Random Forest Regressor

# In[35]:


get_ipython().run_cell_magic('time', '', '# hyperparameter optimization\n\n# define hyperparameters to tune\ngrid = {\n    "n_estimators" : [50, 100],\n    "max_depth" : [None, 2, 10]\n}\n# define the model \nregressor = RandomForestRegressor(random_state = 12345)\n# define the grid search\ngrid_search_rf = GridSearchCV(estimator = regressor, param_grid = grid, scoring="neg_mean_squared_error", cv=5)\n# execute search\ngrid_search_rf.fit(X_train, y_train)\n# summarize result\nprint(\'The best hyperparameters are: {}\'.format(grid_search_rf.best_params_))')


# In[36]:


# function to train model and make predictions
def train_random_forest(X_train, y_train):
    """This function trains a random forest model"""
    global rf_model
    # build the model
    rf_model = RandomForestRegressor(**grid_search_rf.best_params_)
    rf_model.fit(X_train, y_train) # train the model 
    
def random_forest_regressor_prediction(X_test, y_test):
    """
    This function is used to make prediction 
    using a random forest regression model
    """
    rf_pred = rf_model.predict(X_test)
    # determine RMSE for random forest regressor
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    print("\033[1m" + 'RMSE using Random Forest Regressor' + "\033[0m")
    print('RMSE: {:.3f}'.format(rf_rmse))
    print()
    # feature importance from random forest regression 
    sorted_feature_importance = rf_model.feature_importances_.argsort()
    plt.figure(figsize=(8,6))
    plt.barh(X_train.columns[sorted_feature_importance], 
             rf_model.feature_importances_[sorted_feature_importance], 
             color='turquoise')
    plt.xlabel("Random Forest Feature Importance")


# In[37]:


get_ipython().run_cell_magic('time', '', '# train random forest model\ntrain_random_forest(X_train, y_train)')


# In[38]:


get_ipython().run_cell_magic('time', '', '# make predictions with random forest regressor for validation data\nrandom_forest_regressor_prediction(X_valid, y_valid)')


# In[ ]:





# ### Model testing
# 
# The XGBoost regressor is chosen as the model for the final testing since it gave the lowest RMSE and have the lowest tuning and prediction time.

# In[42]:


# make predictions with XGBoost regressor for test data
xgboost_regressor_prediction(X_test, y_test)


# ### Conclusion
# 
# We explored the data and studied the general information about the data. We observed that the data consisted of 100000rows and 11 features. looking through the features we observed that there are 3 categorical features and 7 numberical features. We explored the data by finding the unique count of the `type` features and grouped `type` by the amount to see the amount grouped by type of transaction. We checked for correlation within the data and observed that there is high correlation between `oldbalanceDest` and `newbalanceDest` and also between `oldbalanceOrig` and `newbalanceOrig`. We also observed that `isFlaggedFraud` contains just '0' as its unique value. Since we don't have any more information about the data, we dropped that `isFlaggedFraud` feature. 
# 
# We passed the data through a preprocessing pipeline and split the data into 60% training, 20% testing and 20% validation sets. We trained the `linear regressor`, `KNeighbors regressor`, `lightGBM regressor`, `Random Forest regressor`, and the `XGBoost regressor` on the train set, tuned the model and predicted on the test set. We observed that the `XGBoost regressor` and the `Random Forest regressor` performed better for this data. using the `XGBoost regressor` as the final model, we obtained an RMSE score of 278339.832 on the test set. The `XGBoost regressor` should that the most important factor affecting the predicted amount is the `type`, `step`, and `newbalanceDest`. 
# 
# Assuming time constraint and computational resources is not a factor, dropping either one of `oldbalanceDest` and `newbalanceDest` and either one of `oldbalanceOrig` and `newbalanceOrig` could lead to improved model performance since either of those features are highly correlated to one another. 
