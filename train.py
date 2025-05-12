import warnings

import joblib
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.exceptions import ConvergenceWarning

from data_utils import load_data
from evaluate import evaluate_classification, evaluate_clustering
from model_utils import get_model

# 1. Classification experiments


def run_classification(dataset: str, models: list[str], dump_best: bool = False):
    X_train, X_test, y_train, y_test, _ = load_data(dataset)
    best_model = None
    best_acc = 0.0
    for name in models:
        model = get_model(name)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"{name}: accuracy={acc:.3f}")
        evaluate_classification(model, X_test, y_test)
        # track best
        if dump_best and acc > best_acc:
            best_acc = acc
            best_model = model
    # dump best model if requested
    if dump_best and best_model is not None:
        joblib.dump(best_model, "best_model.pkl")
        print(f"Saved best_model.pkl with accuracy={best_acc:.3f}")


# 2. Clustering on BRCA (attempt OpenML, fallback to builtin Breast Cancer)


def run_clustering():
    try:
        brca = fetch_openml("BRCA", version=1)
        X = brca.data.select_dtypes("float64").to_numpy()
    except Exception:
        print("Warning: BRCA dataset not found on OpenML; using breast_cancer dataset.")
        data = load_breast_cancer()
        X = data.data
    for k in (2, 3, 4, 5):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X)
        evaluate_clustering(X, labels, title=f"KMeans k={k}")


if __name__ == "__main__":
    # run and dump best model on breast_cancer
    run_classification("wine", ["logistic", "svm", "tree"], dump_best=False)
    run_classification(
        "breast_cancer", ["logistic", "svm", "tree", "forest"], dump_best=True
    )
    run_clustering()
