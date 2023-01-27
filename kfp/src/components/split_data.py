from kfp.v2.dsl import Dataset, Input, Output, component

from utils.dependencies import PANDAS, PYARROW, PYTHON310


@component(base_image=PYTHON310, packages_to_install=[PANDAS, PYARROW])
def split_data(train_data_size: float, processed_data: Input[Dataset],
               train_data: Output[Dataset],
               test_data: Output[Dataset]) -> None:
    """
    Split processed data into train and test data.
    
    Args:
        train_data_size (float): Train-test split
        processed_data (Input[Dataset]): Processed dataset
        train_data (Output[Dataset]): Train dataset
        test_data (Output[Dataset]): Test dataset
    """
    import pandas as pd

    processed_df = pd.read_parquet(processed_data.path + '.parquet')
    train_df = processed_df.loc[:int(len(processed_df) * train_data_size)]
    test_df = processed_df.loc[int(len(processed_df) * train_data_size):]
    train_df.to_parquet(train_data.path + '.parquet', index=False)
    test_df.to_parquet(test_data.path + '.parquet', index=False)
