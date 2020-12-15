import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def get_preprocessor(X, categorical_features):
    """
    The numeric data is standard-scaled after mean-imputation,
    while the categorical data is one-hot encoded after imputing missing values
    with a new category ('missing').
    """
    if isinstance(X, pd.DataFrame):
        numeric_features = list(set(X.columns) - set(categorical_features))
    else:
        numeric_features = list(set(range(X.shape[1])) - set(categorical_features))

    categorical_transformer = Pipeline(
        steps=[
            # ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            # ('imputer', SimpleImputer(strategy='median')),
            ("scaler", StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        n_jobs=-1,
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
    )

    return preprocessor


def sparse_to_df(sparse_mat):
    return pd.DataFrame.sparse.from_spmatrix(sparse_mat)


def get_noise_features(X, categorical_features):
    preprocessor = get_preprocessor(X, categorical_features)
    x_preprocessed = preprocessor.fit_transform(X)
    return x_preprocessed.shape[1]


def gen_random_noise(shape):
    mu = 0
    sigma = 1
    z = sigma * np.random.randn(*shape) + mu
    return pd.DataFrame(z)


def plot_losses(hist, title):
    plt.figure(figsize=(10, 5))
    plt.grid()
    plt.title(title, fontsize=25)
    plt.plot(hist["loss_g"], "-o", label="loss_g", linewidth=2.0)
    plt.plot(hist["loss_bb"], "-o", label="loss_bb", linewidth=2.0)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Number of epochs", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.legend(loc="upper right", fontsize=10)
    plt.show()
