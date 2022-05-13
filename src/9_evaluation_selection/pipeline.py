#models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#tube
from sklearn.pipeline import Pipeline
#scaler
from sklearn.preprocessing import StandardScaler

def get_models():
    models = list()
    models.append(LogisticRegression())
    models.append(KNeighborsClassifier())
    models.append(DecisionTreeClassifier())
    return models


def create_pipeline_logisticregression(
    use_scaler: bool, max_iter: int, logreg_C: float, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

    pipeline_steps.append(
        (
            "classifier",
            LogisticRegression(
                random_state=random_state, max_iter=max_iter, C=logreg_C
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)

def create_pipeline_kneighborsclassifier(
    n_neighbors:int, weights: str, algorithm: str, n_jobs: int
) -> Pipeline:
    pipeline_steps = []

    pipeline_steps.append(("scaler", StandardScaler()))

    pipeline_steps.append(
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, n_jobs=n_jobs
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)