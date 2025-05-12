from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin

MODEL_REGISTRY = {
    'logistic': lambda: LogisticRegression(max_iter=10000),
    'svm':      lambda: SVC(probability=True),
    'tree':     lambda: DecisionTreeClassifier(random_state=42),
    'forest':   lambda: RandomForestClassifier(random_state=42),
}

def get_model(name: str) -> ClassifierMixin:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found.")
    return MODEL_REGISTRY[name]()