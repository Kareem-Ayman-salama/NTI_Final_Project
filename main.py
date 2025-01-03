from src.load_data import load_data
from src.Join import join_and_prepare
from src.preprocess import preprocess_final_table
from src.train_model import train_model
from src.evaluate_model import evaluate_model, plot_confusion_matrix, plot_roc_curve

def main():
    # Load datasets
    print("Loading datasets...")
    datasets = load_data("data/")
    application_train = datasets["application_train"]
    installments = datasets["installments_payments"]
    credit_card = datasets["credit_card_balance"]
    pos_cash = datasets["pos_cash_balance"]
    previous_app = datasets["previous_application"]

    # Join and prepare data
    print("Joining and preparing data...")
    final_table = join_and_prepare(previous_app, installments, credit_card, pos_cash)

    # Preprocess the final table
    print("Preprocessing data...")
    preprocessed_data = preprocess_final_table(final_table)

    # Split features and target
    X = preprocessed_data.drop(columns=["TARGET"])
    y = preprocessed_data["TARGET"]

    # Train model
    print("Training model...")
    model, X_train, X_test, y_train, y_test = train_model(X, y, model_type="xgboost", scale_pos_weight=10)

    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(metrics["confusion_matrix"], labels=["Class 0", "Class 1"])
    plot_roc_curve(model, X_test, y_test)

if __name__ == "__main__":
    main()
