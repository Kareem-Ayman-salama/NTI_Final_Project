import streamlit as st 
import pandas as pd
from src.load_data import load_data
from src.Join import join_and_prepare 
from src.preprocess import preprocess_final_table 
from src.train_model import train_model
from src.evaluate_model import evaluate_model, plot_roc_curve

st.title("Home Credit Default Risk")

# Upload data files
st.sidebar.header("Upload Data")
uploaded_files = st.sidebar.file_uploader("Upload CSV files", accept_multiple_files=True)

if uploaded_files:
    st.sidebar.success("Files uploaded successfully!")

    # Process uploaded files
    data_dict = {file.name: pd.read_csv(file) for file in uploaded_files}

    # Join, preprocess, and train model
    st.write("Processing data...")
    application_train = data_dict["application_train.csv"]
    installments = data_dict["installments_payments.csv"]
    credit_card = data_dict["credit_card_balance.csv"]
    pos_cash = data_dict["POS_CASH_balance.csv"]
    previous_app = data_dict["previous_application.csv"]

    final_table = join_and_prepare(previous_app, installments, credit_card, pos_cash)
    preprocessed_data = preprocess_final_table(final_table)

    X = preprocessed_data.drop(columns=["TARGET"])
    y = preprocessed_data["TARGET"]

    model, X_train, X_test, y_train, y_test = train_model(X, y, model_type="xgboost", scale_pos_weight=10)

    metrics = evaluate_model(model, X_test, y_test)
    st.write(metrics)
    plot_roc_curve(model, X_test, y_test)
