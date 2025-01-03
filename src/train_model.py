from sklearn.model_selection import train_test_split # type: ignore
from xgboost import XGBClassifier # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore

def train_model(X, y, model_type="xgboost", test_size=0.2, random_state=42, **kwargs):
    """
    Train a machine learning model with the specified type and parameters.

    Args:
        X (DataFrame): Features for training.
        y (Series): Target variable.
        model_type (str): Type of model to train ("xgboost", "random_forest", "logistic_regression").
        test_size (float): Proportion of the data to include in the test split.
        random_state (int): Random seed for reproducibility.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        tuple: Trained model, X_train, X_test, y_train, y_test
    """
    # Split the data into training and testing sets
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Choose the model based on the specified type
    print(f"Training {model_type} model...")
    if model_type == "xgboost":
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state, **kwargs)
    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=random_state, **kwargs)
    elif model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Train the model
    model.fit(X_train, y_train)
    print(f"{model_type} model training complete!")

    return model, X_train, X_test, y_train, y_test

