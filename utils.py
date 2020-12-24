import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


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


def get_scaler(preprocessor):
    scaler = preprocessor.transformers_[0][1].named_steps['scaler']
    return scaler


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


def calc_similarities(gen_data_above_c, X_train):
    """
    Calculate cosine similarities between the 
    generated samples above confidence level c, and the training data.

     Args:
         X_train:
            training data with shape (N, D)
        gen_data_above_c:
            generated samples above confidence level c with shape (G, D)

    Output:
        Dict that maps: gen_sample_above_c -> (most_similiar_sample_x_train, cosine_score)

    """
    cosine_scores = {}

    # X_train.shape = (N, D)
    for index, row in gen_data_above_c.iterrows():
        row_np = row.to_numpy().reshape(1, -1)
        # row_np.shape = (1, D)
        d_cosine = cosine_similarity(row_np, X_train).squeeze()
        # d.shape = (1, N)
        most_similar_sample_idx = np.argmax(d_cosine)
        similarity_score = round(d_cosine[most_similar_sample_idx], 3)
        cosine_scores[index] = (most_similar_sample_idx, similarity_score)
        
    return cosine_scores


def plot_confidence_levels(y_conf_gen, fig_title):
    counts = pd.value_counts(y_conf_gen, bins=10, sort=False)
    plt.figure()
    ax = counts.plot.bar(rot=0, grid=True, color='#607c8e', figsize=(15,5))
    ax.set_xticklabels([str(interval) for interval in counts.index], fontsize=11)
    ax.set_ylabel('Frequency', fontsize=15)

    ax.set_title(fig_title, fontsize=25)
    plt.show()