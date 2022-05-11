# 9_evaluation_selection

## Usage
This package allows you to train model for predict to predict the forest cover type.
2. Download dataset [Forest]: https://www.kaggle.com/competitions/forest-cover-type-prediction/overview
3. I used Python 3.8 (i cant install poetry with Python 3.9) and [Poetry](https://python-poetry.org/docs/) 
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```

1 - Spruce/Fir;
2 - Lodgepole Pine;
3 - Ponderosa Pine;
4 - Cottonwood/Willow;
5 - Aspen;
6 - Douglas-fir;
7 - Krummholz;
