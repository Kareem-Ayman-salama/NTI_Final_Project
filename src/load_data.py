import pandas as pd

def load_data():
    """
    load_data function loads all datasets from the data folder and returns them as a dictionary.
    """
    # Load all datasets
    application_train = pd.read_csv("data/application_train.csv")
    application_test = pd.read_csv("data/application_test.csv")
    bureau = pd.read_csv("data/bureau.csv")
    bureau_balance = pd.read_csv("data/bureau_balance.csv")
    previous_application = pd.read_csv("data/previous_application.csv")
    pos_cash_balance = pd.read_csv("data/POS_CASH_balance.csv")
    installments_payments = pd.read_csv("data/installments_payments.csv")
    credit_card_balance = pd.read_csv("data/credit_card_balance.csv")

    print("Datasets loaded successfully!")

    return {
        "application_train": application_train,
        "application_test": application_test,
        "bureau": bureau,
        "bureau_balance": bureau_balance,
        "previous_application": previous_application,
        "pos_cash_balance": pos_cash_balance,
        "installments_payments": installments_payments,
        "credit_card_balance": credit_card_balance
    }
