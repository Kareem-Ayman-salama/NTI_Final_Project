import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_competition(competition_name, save_path="data/"):
    """
    Download a competition dataset from Kaggle using the Kaggle API.

    Args:
        competition_name (str): The Kaggle competition identifier (e.g., "home-credit-default-risk").
        save_path (str): Path to save the downloaded dataset. Default is "data/".

    Returns:
        None
    """
    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Download the competition dataset
    print(f"Downloading competition data for {competition_name}...")
    api.competition_download_files(competition_name, path=save_path, unzip=True)
    print(f"Competition data downloaded and saved to {save_path}")

# Example usage:
# Replace "home-credit-default-risk" with the competition name
download_kaggle_competition("home-credit-default-risk")
