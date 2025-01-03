import pandas as pd
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.preprocessing import MinMaxScaler, LabelEncoder # type: ignore
from src.aggregation import preprocess_final_table
def handle_missing_values(data, strategy="median"):
    """
    Handle missing values in the dataset using the specified strategy.

    Args:
        data (DataFrame): The input dataset.
        strategy (str): Strategy to fill missing values (default is "median").
    
    Returns:
        DataFrame: Dataset with missing values handled.
    """
    imputer = SimpleImputer(strategy=strategy)
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
    data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
    return data

def encode_categorical_features(data):
    """
    Encode categorical features using Label Encoding and One-Hot Encoding.

    Args:
        data (DataFrame): The input dataset.
    
    Returns:
        DataFrame: Dataset with categorical features encoded.
    """
    # Label Encoding for binary categorical variables
    label_encoder = LabelEncoder()
    binary_columns = [col for col in data.select_dtypes(include=["object"]).columns if data[col].nunique() == 2]
    for col in binary_columns:
        data[col] = label_encoder.fit_transform(data[col])

    # One-Hot Encoding for remaining categorical variables
    data = pd.get_dummies(data, columns=[col for col in data.select_dtypes(include=["object"]).columns if col not in binary_columns], drop_first=True)
    return data

def scale_features(data):
    """
    Scale numeric features to a range between 0 and 1.

    Args:
        data (DataFrame): The input dataset.
    
    Returns:
        DataFrame: Dataset with scaled features.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data

def preprocess_final_table(final_table):
    """
    Preprocess the final aggregated and joined table for modeling.
    
    Steps:
    - Handle missing values.
    - Encode categorical features.
    - Scale numeric features.
    
    Args:
        final_table (DataFrame): The final table after join and aggregation.
    
    Returns:
        DataFrame: Preprocessed table ready for modeling.
    """
    # Handle missing values
    print("Handling missing values...")
    final_table = handle_missing_values(final_table)

    # Encode categorical features
    print("Encoding categorical features...")
    final_table = encode_categorical_features(final_table)

    # Scale numeric features
    print("Scaling numeric features...")
    final_table = scale_features(final_table)

    print("Preprocessing complete!")
    return final_table



# preprocessed_data = preprocess_final_table(final_table)
