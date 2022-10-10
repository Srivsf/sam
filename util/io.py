"""Input/Output operations for reading and writing objects between S3 and 
other media.
"""

import os
from io import BytesIO

import joblib
import boto3
import pandas as pd

from dhi.dsmatch import s3_ds_bucket
from dhi.dsmatch import local_bucket

def read_csv(filepath: str, bucket: str=local_bucket, 
             low_memory=False, **kwargs):
    """Read a CSV file into a Pandas DataFrame. Try reading from SageMaker 
    first, and if not found pull from S3.

    Args:
        filepath (str): path to the csv file, possibly excluding the 
        root/bucket.
        bucket (str, optional): Root/Bucket. Defaults to local_bucket.

    Returns:
        pd.DataFrame: The CSV file as a Pandas DataFrame.
    """
    global s3_ds_bucket
    splitted = filepath.split(bucket)
    if len(splitted) > 1:
        filepath = splitted[-1][1:]
    splitted = filepath.split(s3_ds_bucket)
    if len(splitted) > 1:
        filepath = splitted[-1][1:]
   
    try:
        return pd.read_csv(os.path.join(bucket, filepath), 
                low_memory=low_memory, **kwargs)
    except FileNotFoundError:
        pass
    return pd.read_csv(os.path.join('s3://', s3_ds_bucket, filepath), 
            low_memory=low_memory, **kwargs)

def load_joblib(bucket=s3_ds_bucket, key=None):
    if key is None:
        raise ValueError('key cannot be None.')
    # https://stackoverflow.com/a/59903472/394430
    with BytesIO() as data:
        boto3.resource('s3').Bucket(bucket).download_fileobj(key, data)
        data.seek(0)    # move back to the beginning after writing
        return joblib.load(data)
