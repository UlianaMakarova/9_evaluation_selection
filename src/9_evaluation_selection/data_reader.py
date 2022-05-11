from pathlib import Path
from typing import Tuple
import click

import pandas as pd

from sklearn.model_selection import train_test_split

def get_dataset(
    csv_path: Path, random_state: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset_train = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset_train.shape}.")
    features = dataset_train.drop("Cover_Type", axis=1)
    target = dataset_train["Cover_Type"]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_split_ratio, random_state=random_state
    )
    return features_train, target_train, features_val, target_val
