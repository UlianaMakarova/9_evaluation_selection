from pathlib import Path
from typing import Tuple
import click

import pandas as pd

def get_dataset(
    csv_path: Path, random_state: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset_train = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset_train.shape}.")
    features_train = dataset_train.drop("Cover_Type", axis=1)
    target_train = dataset_train["Cover_Type"]
    return features_train, target_train
