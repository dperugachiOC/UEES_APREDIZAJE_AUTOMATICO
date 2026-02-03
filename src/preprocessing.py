import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_and_preprocess_data(
    data_path="DATA/Social_Network_Ads.csv",
    test_size=0.2,
    random_state=42
):
    """
    Carga el dataset, aplica preprocesamiento y divide en train/test.

    Returns:
        X_train, X_test, y_train, y_test
        preprocess_scaled, preprocess_no_scale
    """

    # 1. Cargar datos
    df = pd.read_csv(data_path)

    # 2. Variable objetivo
    y = df["Purchased"]

    # 3. Variables predictoras
    X = df.drop(columns=["Purchased", "User ID"])

    # 4. Definir tipos de columnas
    categorical_features = ["Gender"]
    numerical_features = ["Age", "EstimatedSalary"]

    # 5. Preprocesadores
    preprocess_scaled = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical_features),
        ]
    )

    preprocess_no_scale = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical_features),
        ]
    )

    # 6. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocess_scaled,
        preprocess_no_scale,
    )
