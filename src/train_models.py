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
        "model__kernel": ["linear", "rbf"],
        "model__C": [0.1, 1, 10, 100],
        "model__gamma": ["scale", "auto"]
    }

    svm = GridSearchCV(
        svm_pipeline,
        svm_params,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    # 3. Random Forest
    random_forest = Pipeline(steps=[
        ("preprocess", preprocess_no_scale),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced_subsample"
        ))
    ])

    # 4. Regresión Logística
    logistic_regression = Pipeline(steps=[
        ("preprocess", preprocess_scaled),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    models = {
        "Decision Tree": decision_tree,
        "SVM": svm,
        "Random Forest": random_forest,
        "Logistic Regression": logistic_regression
    }

    # Entrenamiento
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Modelo entrenado: {name}")

    return models
