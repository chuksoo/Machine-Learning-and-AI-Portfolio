import joblib
import streamlit as st
import pandas as pd
import numpy as np
from predictions import predict

# Page config
st.set_page_config(page_title="Card Transactions",
                   page_icon="images/transactions_icon.png", 
                   layout="wide")

st.title("Credit Card Transactions Fraud Detection")
st.image("images/fraud_detection.png")
st.write("This is a credit card fraud detection app.")
st.markdown("This app uses a machine learning model to predict whether a given transaction will be fraudulent or not.")

# Sidebar
st.sidebar.title("Card Transactions Fraud Detection")
st.sidebar.image("images/get_help.png")
st.sidebar.write("This is a credit card fraud detection app.")
st.sidebar.write("This app uses a machine learning model to predict whether a given transaction will be fraudulent or not.")
st.sidebar.write("The model is trained on a dataset of credit card transactions.")

# Load the model
with open('./streamlit_app/transactions-deploy/transactions_lgbm_model.sav', 'rb') as model_file:
    model = joblib.load(model_file)

# Streamlit interface to input data
col1, col2 = st.columns(2)

with col1:
    availableMoney = st.number_input(label="Available Money")
    transactionAmount = st.number_input(label="Transaction Amount")
    merchantName = st.number_input(label="Merchant Name")
    merchantCategoryCode = st.number_input(label="Merchant Category Code")
    accountNumber = st.number_input(label="Account Number")
    currentBalance = st.number_input(label="Current Balance")
    accountOpenMonth = st.number_input(label="Account Open Month")
    accountOpenDayofWeek = st.number_input(label="Account Open Day of Week")
    creditLimit = st.number_input(label="Credit Limit")
    currentExpMonth = st.number_input(label="Current Expiration Month")

with col2:
    posEntryMode = st.number_input(label="Pos Entry Mode")
    transactionMonth = st.number_input(label="Transaction Month")
    transactionHour = st.number_input(label="Transaction Hour")
    transactionMinutes = st.number_input(label="Transaction Minutes")
    transactionSeconds = st.number_input(label="Transaction Seconds")
    posConditionCode = st.number_input(label="Pos Condition Code")
    posEntryMode = st.number_input(label="Pos Entry Mode")
    cardPresent = st.number_input(label="Card Present")
    dateOfLastAddressChangeMonth = st.number_input(label="Month of last address change")
    dateOfLastAddressChangeDayofWeek = st.number_input(label="Day of week of last address change")

    # Streamlit interface to input data
    def prediction(availableMoney, merchantName, merchantCategoryCode, accountNumber, currentBalance, accountOpenMonth, accountOpenDayofWeek, 
                   creditLimit, currentExpMonth, customerId, posEntryMode, transactionMonth, transactionHour, transactionMinutes, 
                   transactionSeconds, posConditionCode, cardPresent, dateOfLastAddressChangeMonth, dateOfLastAddressChangeDayofWeek):
        # create a df with input
        df_input = pd.DataFrame({
                'availableMoney': availableMoney,
                'transactionAmount': transactionAmount,
                'merchantName': merchantName,
                'merchantCategoryCode': merchantCategoryCode,
                'accountNumber': accountNumber,
                'currentBalance': currentBalance,
                'accountOpenMonth': accountOpenMonth,
                'accountOpenDayofWeek': accountOpenDayofWeek,
                'creditLimit': creditLimit,
                'currentExpMonth': currentExpMonth,
                'customerId': customerId,
                'posEntryMode': posEntryMode,
                'transactionMonth': transactionMonth,
                'transactionHour': transactionHour,
                'transactionMinutes': transactionMinutes,
                'transactionSeconds': transactionSeconds,
                'posConditionCode': posConditionCode,
                'cardPresent': cardPresent,
                'dateOfLastAddressChangeMonth': dateOfLastAddressChangeMonth,
                'dateOfLastAddressChangeDayofWeek': dateOfLastAddressChangeDayofWeek
        })

        prediction = model.predict(df_input)
        return prediction
    
    # Predict button
    if st.button('Predict'):
        predict = prediction(availableMoney, merchantName, merchantCategoryCode, accountNumber, currentBalance, accountOpenMonth, accountOpenDayofWeek, 
                   creditLimit, currentExpMonth, customerId, posEntryMode, transactionMonth, transactionHour, transactionMinutes, 
                   transactionSeconds, posConditionCode, cardPresent, dateOfLastAddressChangeMonth, dateOfLastAddressChangeDayofWeek)
        if predict == 0:
            st.write("This transaction is not fraudulent.")
        else:
            st.write("This transaction is fraudulent.")
            