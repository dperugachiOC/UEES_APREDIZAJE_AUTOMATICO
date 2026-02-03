from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def train_all_models(X_train, y_train, preprocess_scaled, preprocess_no_scale):
    """
    Entrena 4 modelos supervisados y los devuelve en un diccionario.
    """

    # 1. Árbol de Decisión
    decision_tree = Pipeline(steps=[
        ("preprocess", preprocess_no_scale),
        ("model", DecisionTreeClassifier(random_state=42))
    ])

    # 2. SVM con GridSearch
    svm_pipeline = Pipeline(steps=[
        ("preprocess", preprocess_scaled),
        ("model", SVC())
    ])

    svm_params = {
        "model__kernel": ["linear", "]()
