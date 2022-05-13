from pathlib import Path
from typing import Tuple
import click

import pandas as pd

from sklearn.model_selection import train_test_split

def get_dataset(
    csv_path: Path, split_need: bool, random_state: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    target = None
    if "Cover_Type" in dataset:
        features = dataset.drop("Cover_Type", axis=1)
        target = dataset["Cover_Type"]
    else:
        features = dataset

    if split_need:
        features_train, features_val, target_train, target_val = train_test_split(
            features, target,
            test_size=test_split_ratio,
            random_state=random_state
            )
        return features_train, target_train, features_val, target_val
    else:
        return features, target
