from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    silhouette_score
)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Classification metrics

def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"== {model.__class__.__name__} ==")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

# Clustering metrics & plots

def evaluate_clustering(X, labels, title="Clustering"):  # silhouette + PCA scatter
    score = silhouette_score(X, labels)
    print(f"Silhouette Score: {score:.3f}")
    pca = PCA(2)
    proj = pca.fit_transform(X)
    plt.figure()
    plt.scatter(proj[:,0], proj[:,1], c=labels)
    plt.title(f"{title} (silhouette={score:.3f})")
    plt.show()