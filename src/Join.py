from src.aggregation import (
    aggregate_installments,
    aggregate_credit_card,
    aggregate_pos_cash
)
import pandas as pd

def join_and_prepare(previous_application, installments_payments, credit_card_balance, pos_cash_balance):
    """
    Join aggregated data from installments_payments, credit_card_balance,
    and pos_cash_balance with the previous_application table.
    """
    # Step 1: Perform aggregations
    installments_agg = aggregate_installments(installments_payments)
    credit_card_agg = aggregate_credit_card(credit_card_balance)
    pos_cash_agg = aggregate_pos_cash(pos_cash_balance)

    # Step 2: Join aggregated data with previous_application
    previous_application = previous_application.merge(
        installments_agg, on="SK_ID_CURR", how="left"
    )
    previous_application = previous_application.merge(
        credit_card_agg, on="SK_ID_CURR", how="left"
    )
    previous_application = previous_application.merge(
        pos_cash_agg, on="SK_ID_CURR", how="left"
    )

    
    return previous_application
