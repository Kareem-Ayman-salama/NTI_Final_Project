import pandas as pd

def aggregate_pos_cash(pos_cash_balance):
    """
    Perform aggregation on POS_CASH_balance dataset and return aggregated DataFrame.
    
    Args:
        pos_cash_balance (DataFrame): The POS_CASH_balance dataset.
    
    Returns:
        DataFrame: Aggregated POS_CASH_balance dataset.
    """
    pos_cash_agg = pos_cash_balance.groupby("SK_ID_CURR").agg({
        "MONTHS_BALANCE": ["count", "min", "max"],
        "CNT_INSTALMENT": ["mean", "sum", "max"],
        "CNT_INSTALMENT_FUTURE": ["mean", "sum", "max"],
        "SK_DPD": ["mean", "sum", "max"],
        "SK_DPD_DEF": ["mean", "sum", "max"]
    }).reset_index()

    # Rename columns
    pos_cash_agg.columns = ["_".join(col).strip() if col[1] else col[0] for col in pos_cash_agg.columns]
    return pos_cash_agg


def aggregate_installments(installments_payments):
    """
    Perform aggregation on installments_payments dataset and return aggregated DataFrame.
    
    Args:
        installments_payments (DataFrame): The installments_payments dataset.
    
    Returns:
        DataFrame: Aggregated installments_payments dataset.
    """
    # Create additional features: payment difference and delay
    installments_payments["PAYMENT_DIFF"] = installments_payments["AMT_PAYMENT"] - installments_payments["AMT_INSTALMENT"]
    installments_payments["PAYMENT_DELAY"] = installments_payments["DAYS_ENTRY_PAYMENT"] - installments_payments["DAYS_INSTALMENT"]

    # Perform aggregation
    installments_agg = installments_payments.groupby("SK_ID_CURR").agg({
        "AMT_INSTALMENT": ["sum", "mean", "max", "min"],
        "AMT_PAYMENT": ["sum", "mean", "max", "min"],
        "PAYMENT_DIFF": ["mean", "sum", "max", "min"],
        "PAYMENT_DELAY": ["mean", "sum", "max", "min"]
    }).reset_index()

    # Rename columns
    installments_agg.columns = ["_".join(col).strip() if col[1] else col[0] for col in installments_agg.columns]

    return installments_agg


def aggregate_credit_card(credit_card_balance):
    """
    Perform aggregation on credit_card_balance dataset and return aggregated DataFrame.
    
    Args:
        credit_card_balance (DataFrame): The credit_card_balance dataset.
    
    Returns:
        DataFrame: Aggregated credit_card_balance dataset.
    """
    # Perform aggregation
    credit_card_agg = credit_card_balance.groupby("SK_ID_CURR").agg({
        "AMT_BALANCE": ["mean", "sum", "max"],
        "AMT_CREDIT_LIMIT_ACTUAL": ["mean", "max"],
        "AMT_DRAWINGS_ATM_CURRENT": ["mean", "sum", "max"],
        "SK_DPD": ["mean", "sum", "max"],
        "SK_DPD_DEF": ["mean", "sum", "max"]
    }).reset_index()

    # Rename columns
    credit_card_agg.columns = ["_".join(col).strip() if col[1] else col[0] for col in credit_card_agg.columns]

    return credit_card_agg


def aggregate_previous_application(previous_application):
    """
    Perform aggregation on previous_application dataset and return aggregated DataFrame.
    
    Args:
        previous_application (DataFrame): The previous_application dataset.
    
    Returns:
        DataFrame: Aggregated previous_application dataset.
    """
    # Create additional features: approval rate and days since last decision
    previous_application["APPROVAL_RATE"] = previous_application["AMT_CREDIT"] / previous_application["AMT_APPLICATION"]
    previous_application["APPROVAL_RATE"].fillna(0, inplace=True)

    # Perform aggregation
    previous_agg = previous_application.groupby("SK_ID_CURR").agg({
        "AMT_APPLICATION": ["mean", "sum", "max", "min"],
        "AMT_CREDIT": ["mean", "sum", "max", "min"],
        "APPROVAL_RATE": ["mean", "max"],
        "NAME_CONTRACT_STATUS": ["nunique"],  # Number of unique contract statuses
        "CNT_PAYMENT": ["mean", "sum", "max"],
        "DAYS_DECISION": ["max", "min"],  # Most recent and oldest decision
    }).reset_index()

    # Rename columns
    previous_agg.columns = ["_".join(col).strip() if col[1] else col[0] for col in previous_agg.columns]

    print("Previous_application aggregation complete!")
    return previous_agg
