import pandas as pd
from src.utils.logger import get_logger, configure_logging

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    logger = get_logger(__name__)

    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)

    if df is None:
        logger.error(f"Failed to load data from {file_path}")
    elif df.empty:
        logger.warning(f"Loaded empty DataFrame from {file_path}")
    else:
        logger.info(f"Data loaded successfully from {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")

    return df