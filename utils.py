import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

from collections import defaultdict



DATA_PATH = "./data/"
ToEncodeLabelDatasets = ["ailerons", "diabetes", "wind", "adult"]


def get_preprocessor(X, categorical_features):
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
    scaler = preprocessor.transformers_[0][1].named_steps["scaler"]
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
    #print("count\n")
    #print(counts)
    plt.figure()
    ax = counts.plot.bar(rot=0, grid=True, color="#607c8e", figsize=(15, 5))
    #print("ax\n")
    #print(ax)
    ax.set_xticklabels([str(interval) for interval in counts.index], fontsize=11)
    ax.set_ylabel("Frequency", fontsize=15)

    ax.set_title(fig_title, fontsize=25)
    plt.show()


def plot_similarities_dist(gen_data_above_c, X_train):
    for index, row in gen_data_above_c.iterrows():
        row_np = row.to_numpy().reshape(1, -1)
        d_cosine = cosine_similarity(row_np, X_train).squeeze()

        plt.figure(figsize=(15, 5))
        plt.hist(d_cosine, color="#607c8e", edgecolor="black", bins=35)
        plt.title(f"Similarities Distribution (sample {index})", fontsize=25)
        plt.ylabel("Frequency", fontsize=15)
        plt.xlabel("Similarity", fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.show()
        print("\n")


def read_data(data_name, data=None):
    """
    read data and label encode the labels
    :param data_name: dataset name (from data folder)
    :param data: data which already loaded using pd.read_csv (adult for example)
    :return: X, y (data and labels)
    """
    if data is None:
        data = pd.read_csv(f"{DATA_PATH}/{data_name}.csv").to_numpy()
    X = data[:, :-1]
    y = data[:, -1:].squeeze()
    label_encoder = None

    if data_name in ToEncodeLabelDatasets:
        label_encoder = preprocessing.LabelEncoder()
        y = label_encoder.fit_transform(y)

    if len(np.unique(y)) != 2:
        raise ValueError(f"'{data_name}' is not binary classification")
    return X, y, label_encoder


def calc_coverage(gen_data, X_train, sim_threshold, conf_diff_threshold, y_conf_gen, y_conf_train):
    count = 0

    for index, row in X_train.iterrows():
        row_np = row.to_numpy().reshape(1, -1)
        # row_np.shape = (1, D)
        d_cosine = cosine_similarity(row_np, gen_data).squeeze()
        # d.shape = (1, N)
        ind = np.nonzero(d_cosine >= sim_threshold)[0]
        if len(ind) == 0: # no samples with similarity greater then sim_threshold
            continue

        conf_diff = y_conf_gen[ind] - y_conf_train[ind]
        if np.any(np.abs(conf_diff) <= conf_diff_threshold):
            count += 1

    coverage = (count / X_train.shape[0]) * 100
    return round(coverage, 4)


def calc_precision(gen_data, X_train, sim_threshold, conf_diff_threshold, y_conf_gen, y_conf_train):
    count = 0

    for index, row in gen_data.iterrows():
        row_np = row.to_numpy().reshape(1, -1)
        # row_np.shape = (1, D)
        d_cosine = cosine_similarity(row_np, X_train).squeeze()
        # d.shape = (1, N)
        ind = np.nonzero(d_cosine >= sim_threshold)[0]
        if len(ind) == 0: # no samples with similarity greater then sim_threshold
            continue

        conf_diff = y_conf_gen[ind] - y_conf_train[ind]
        if np.any(np.abs(conf_diff) <= conf_diff_threshold):
            count += 1

    precision = (count / gen_data.shape[0]) * 100
    return round(precision, 4)


def table(gen_data, X_train, y_conf_gen, y_conf_train):
    similarity_thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]
    conf_diff_thresholds = [0.01, 0.05, 0.1, 0.15, 0.2]
    data = defaultdict(list)
    data2 = defaultdict(list)
    data3 = defaultdict(list)

    for sim_threshold in similarity_thresholds:
        for conf_diff_threshold in conf_diff_thresholds:
            coverage = calc_coverage(gen_data, X_train, sim_threshold, conf_diff_threshold, y_conf_gen, y_conf_train)
            precision = calc_precision(gen_data, X_train, sim_threshold, conf_diff_threshold, y_conf_gen, y_conf_train)
            data[sim_threshold].append(f"{coverage} | {precision}")
            data2[sim_threshold].append(coverage)
            data3[sim_threshold].append(precision)

    results = pd.DataFrame.from_dict(data, orient='index', columns=conf_diff_thresholds)
    coverage= pd.DataFrame.from_dict(data2, orient='index', columns=conf_diff_thresholds)
    precision = pd.DataFrame.from_dict(data3, orient='index', columns=conf_diff_thresholds)
    return results, coverage, precision



def gen_data_to_same_conf_dist_as_train(y_conf_gen, y_conf_train):
    """
    generate samples until ..you have the same number of samples as those
    of the training set and in the same confidence distribution
    """
    train_bucktes = pd.value_counts(y_conf_train, bins=10, sort=False)
    idxs, freqs = train_bucktes.index, train_bucktes.values
    ans = []

    for sample_idx, sample_conf in enumerate(y_conf_gen):

        # find interval index which contains the sample_conf (-1 if not found)
        interval_idx = np.nonzero(idxs.contains(sample_conf))[0]

        # value not fould (empty list of indices)
        if len(interval_idx) == 0:
            continue

        # check if the bucket is not full
        interval_idx = interval_idx[0]
        if freqs[interval_idx] > 0:
            ans.append(sample_idx)
            freqs[interval_idx] -= 1

    return ans
