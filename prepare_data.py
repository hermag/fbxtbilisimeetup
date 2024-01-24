
import argparse
import logging
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Union
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

pd.options.mode.chained_assignment = None  # default='warn'


# ----- CONSTANTS ----- #
# Columns of df
# Features to consider for the model
FEATURES = ['Age','Annual_Income','Monthly_Inhand_Salary','Num_Bank_Accounts','Outstanding_Debt',
            'Credit_Utilization_Ratio','Total_EMI_per_month','Amount_invested_monthly',
            'Monthly_Balance','Credit_Score']
# Filname of the raw data file
RAW_DATA_FILE = 'data.csv'
            
def get_train_test_split(df: pd.DataFrame, n_test_data: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the input data frame into a training and test set.

    Args:
        df: Raw input data.
        n_test_data: Amount of test data, in decimal values, i.e. for 20% it should be 0.2

    Returns:
        Tuple[pd.DataFrame]: Raw train and test data splits.
    """

    df_train, df_test = train_test_split(df, test_size=n_test_data, random_state=42)

    logger.info(f"Train, Test and Validation data sets preparation")
    
    return df_train, df_test

def wrap_transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper for transforming the data for the model
    
    Processing is applied in the following steps:
        moves the last (target) column to the first place
        
    Args:
        df: Input dataframe.
    
    Returns:
        pd.DataFrame: Transformed dataframe.
    """
    columns = df.columns.tolist()
    columns = [columns[-1]] + columns[:-1]
    df = df[columns]
    return df

if __name__ == '__main__':
    logger.info(f'Preprocessing job started.')
    # Parse the SDK arguments that are passed when creating the SKlearn container
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_test_data", type=int, default=10)
    parser.add_argument("--n_val_data", type=int, default=10)
    args, _ = parser.parse_known_args()

    logger.info(f"Received arguments {args}.")

    # Read in data locally in the container
    input_data_path = os.path.join("/opt/ml/processing/input", RAW_DATA_FILE)
    logger.info(f"Reading input data from {input_data_path}")
    # Read raw input data
    df = pd.read_csv(input_data_path)
    logger.info(f"Shape of data is: {df.shape}")

    # ---- Preprocess the data set ----
    logger.info("Split data into training+validation and test set.")
    df_train_valid, df_test = get_train_test_split(df=df, n_test_data=args.n_test_data)

    logger.info("Split training+validation into training and validation set.")
    df_train, df_val = get_train_test_split(df=df_train_valid, n_test_data=args.n_val_data) 

    logger.info("Transforming training data.")
    train = wrap_transform_data(
        df=df_train
    )
    
    logger.info("Transforming validation data.")
    val = wrap_transform_data(
        df=df_val
    )

    logger.info("Transforming test data.")
    test = wrap_transform_data(
        df=df_test
    )
    
    # Create local output directories. These directories live on the container that is spun up.
    try:
        os.makedirs("/opt/ml/processing/train")
        os.makedirs("/opt/ml/processing/validation")
        os.makedirs("/opt/ml/processing/test")
        print("Successfully created directories")
    except Exception as e:
        # if the Processing call already creates these directories (or directory otherwise cannot be created)
        logger.debug(e)
        logger.debug("Could Not Make Directories.")
        pass

    # Save data locally on the container that is spun up.
    try:
        pd.DataFrame(train).to_csv("/opt/ml/processing/train/train.csv", header=False, index=False)
        pd.DataFrame(train).to_csv("/opt/ml/processing/train/train_w_header.csv", header=True, index=False)
        pd.DataFrame(val).to_csv("/opt/ml/processing/validation/val.csv", header=False, index=False)
        pd.DataFrame(val).to_csv("/opt/ml/processing/validation/val_w_header.csv", header=True, index=False)
        pd.DataFrame(test).to_csv("/opt/ml/processing/test/test.csv", header=False, index=False)
        pd.DataFrame(test).to_csv("/opt/ml/processing/test/test_w_header.csv", header=True, index=False)
        logger.info("Files Successfully Written Locally")
    except Exception as e:
        logger.debug("Could Not Write the Files")
        logger.debug(e)
        pass

    logger.info("Finished running processing job")
