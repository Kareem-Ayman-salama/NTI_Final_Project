import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of a trained model using test data.

    Args:
        model: The trained machine learning model.
        X_test (DataFrame): Features for testing.
        y_test (Series): True target values for testing.
    
    Returns:
        dict: A dictionary containing evaluation metrics and results.
    """
    # Generate predictions
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Classification Report
    print("Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    return {
        "classification_report": report,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }

def plot_confusion_matrix(cm, labels=None):
    """
    Plot the confusion matrix.

    Args:
        cm (array): Confusion matrix.
        labels (list, optional): Class labels. Defaults to None.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

def plot_roc_curve(model, X_test, y_test):
    """
    Plot the ROC curve.

    Args:
        model: The trained machine learning model.
        X_test (DataFrame): Features for testing.
        y_test (Series): True target values for testing.
    """
    from sklearn.metrics import roc_curve, auc # type: ignore

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

