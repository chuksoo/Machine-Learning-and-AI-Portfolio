import joblib

def predict(data):
    clf = joblib.load("transactions_lgbm_model.sav")
    return clf.predict(data)