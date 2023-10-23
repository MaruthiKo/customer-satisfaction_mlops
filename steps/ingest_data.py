import logging 

import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting data from the data_path provided.
    """
    def __init__(self, path: str):
        """
        Args:
            data_path: path to the data.
        """
        self.path = path

    def get_data(self):
        """
        Getting the data from the path provided.
        Returns:
            pd.DataFrame: The data in a pandas dataframe.
        """
        logging.info(f"Getting data from {self.path}")
        return pd.read_csv(self.path)
    
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data_path provided.

    Args:
        data_path: Path to the data.
    
    Return:
        pd.DataFrame: The data in a pandas dataframe.
    """
    try:
        data = IngestData(data_path)
        df = data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e
