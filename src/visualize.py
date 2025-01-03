import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import pandas as pd

def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot the top N most important features of a model.

    Args:
        model: Trained model with feature importance attribute.
        feature_names (list): List of feature names.
        top_n (int): Number of top features to plot. Defaults to 10.

    Returns:
        None
    """
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(model, X_test, y_test):
    """
    Plot the ROC curve for a trained model.

    Args:
        model: Trained machine learning model.
        X_test (DataFrame): Test features.
        y_test (Series): True target labels.

    Returns:
        None
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

def plot_confusion_matrix(cm, labels=None):
    """
    Plot the confusion matrix.

    Args:
        cm (array): Confusion matrix.
        labels (list, optional): Class labels. Defaults to None.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(data):
    """
    Plot the correlation matrix of a dataset.

    Args:
        data (DataFrame): Input dataset.

    Returns:
        None
    """
    plt.figure(figsize=(12, 10))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_distribution(data, column, target_column=None, bins=30):
    """
    Plot the distribution of a column, optionally grouped by a target variable.

    Args:
        data (DataFrame): Input dataset.
        column (str): Column to plot the distribution of.
        target_column (str, optional): Target column to group by. Defaults to None.
        bins (int): Number of bins for the histogram.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    if target_column:
        for value in data[target_column].unique():
            sns.histplot(data[data[target_column] == value][column], bins=bins, label=f'{target_column}={value}', kde=True)
        plt.legend()
    else:
        sns.histplot(data[column], bins=bins, kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
