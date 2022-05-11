from pathlib import Path
from joblib import dump

import click

import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score

from .data_reader import get_dataset
from .pipeline import create_pipeline_logisticregression
from .pipeline import create_pipeline_kneighborsclassifier

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/results/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
#split dataset
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
#logisticregression
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=1000,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
#KNeighborsClassifier
@click.option(
    "--n_neighbors",
    default=7,
    type=int,
    show_default=True,
)
@click.option(
    "--weights",
    default='uniform',
    type=str,
    show_default=True,
)

@click.option(
    "--algorithm",
    default='auto',
    type=str,
    show_default=True,
)
@click.option(
    "--n_jobs",
    default=1,
    type=int,
    show_default=True,
)

def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    n_neighbors: int,
    weights: str,
    algorithm: str,
    n_jobs: int,
) -> None:
    features_train, target_train, features_val, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    click.echo(f"Features_train shape: {features_train.shape}.")
    click.echo(f"Target_train shape: {target_train.shape}.")
    with mlflow.start_run():
        pipeline1 = create_pipeline_logisticregression(use_scaler, max_iter, logreg_c, random_state)
        pipeline1.fit(features_train, target_train)
        accuracy_logisticregression = accuracy_score(target_val, pipeline1.predict(features_val))
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_metric("accuracy", accuracy_logisticregression)
        click.echo(f"Accuracy: {accuracy_logisticregression}.")
        dump(pipeline1, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
        pipeline2 = create_pipeline_kneighborsclassifier(n_neighbors, weights, algorithm, n_jobs)
        pipeline2.fit(features_train, target_train)
        accuracy_kneighborsclassifier = accuracy_score(target_val, pipeline2.predict(features_val))
        click.echo(f"Accuracy: {accuracy_kneighborsclassifier}.")



