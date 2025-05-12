from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split

def load_data(name: str, test_size=0.3, random_state: int = 0):
    if name == 'wine':
        data = load_wine()
    elif name == 'breast_cancer':
        data = load_breast_cancer()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    X_train, X_test, y_train, y_test = train_test_split(
        data["data"], data["target"],
        test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, data["feature_names"]