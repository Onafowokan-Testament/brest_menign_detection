from data_utils import load_data
from model_utils import get_model
from evaluate import evaluate_classification, evaluate_clustering
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml

# 1. Classification experiments

def run_classification(dataset: str, models: list[str]):
    X_train, X_test, y_train, y_test, features = load_data(dataset)
    for name in models:
        model = get_model(name)
        model.fit(X_train, y_train)
        evaluate_classification(model, X_test, y_test)

# 2. Clustering on BRCA (OpenML)

def run_clustering():
    brca = fetch_openml('BRCA', version=1)
    X = brca.data.select_dtypes('float64').to_numpy()
    for k in (2, 3, 4, 5):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X)
        evaluate_clustering(X, labels, title=f"KMeans k={k}")

if __name__ == '__main__':
    run_classification('wine', ['logistic', 'svm', 'tree'])
    run_classification('breast_cancer', ['logistic', 'svm', 'tree', 'forest'])
    run_clustering()