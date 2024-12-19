import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts_dir', "train.csv")
    test_data_path: str = os.path.join('artifacts_dir', "test.csv")
    raw_data_path: str = os.path.join('artifacts_dir', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset from CSV
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as DataFrame')

            # Create artifacts directory if it doesn't exist
            artifacts_dir = os.path.dirname(self.ingestion_config.train_data_path)
            if not os.path.exists(artifacts_dir):
                os.makedirs(artifacts_dir)
                logging.info(f"Created artifacts directory at: {artifacts_dir}")

            # Save raw data to specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at: {self.ingestion_config.raw_data_path}")

            # Perform train-test split
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save training and testing sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data ingestion completed successfully")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    logging.info(f"Train data path: {train_data_path}")
    logging.info(f"Test data path: {test_data_path}")