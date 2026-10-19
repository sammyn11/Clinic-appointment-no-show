"""Model builders.

Traditional models come from scikit-learn. The networks are Keras models,
one Sequential and one Functional, and they are fed with tf.data pipelines.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from tensorflow import keras
from tensorflow.keras import layers

from noshow.config import SEED


def set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def logistic(C: float = 1.0, class_weight=None) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=400,
        random_state=SEED,
    )


def random_forest(
    n_estimators: int = 200,
    max_depth: int | None = 12,
    min_samples_leaf: int = 5,
    max_features="sqrt",
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )


def hist_gbm(learning_rate: float, max_depth: int, l2: float, max_iter: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=max_iter,
        min_samples_leaf=20,
        l2_regularization=l2,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=12,
        random_state=SEED,
    )


def linear_svm(C: float = 1.0) -> LinearSVC:
    # dual=False is the usual choice when rows outnumber columns.
    return LinearSVC(
        C=C,
        class_weight="balanced",
        dual=False,
        max_iter=4000,
        random_state=SEED,
    )


def class_weight_dict(y: np.ndarray) -> dict[int, float]:
    y = np.asarray(y).astype(int)
    n = len(y)
    n1 = max(int(y.sum()), 1)
    n0 = max(n - int(y.sum()), 1)
    return {0: n / (2.0 * n0), 1: n / (2.0 * n1)}


def sample_weight_balanced(y: np.ndarray) -> np.ndarray:
    weights = class_weight_dict(y)
    y = np.asarray(y).astype(int)
    return np.where(y == 1, weights[1], weights[0]).astype(np.float32)


def make_tfdata(X, y, batch: int, shuffle: bool, seed: int = SEED):
    ds = tf.data.Dataset.from_tensor_slices(
        (X.astype(np.float32), y.astype(np.float32))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 20000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def make_tfdata_embed(X, neigh, y, batch: int, shuffle: bool, seed: int = SEED):
    ds = tf.data.Dataset.from_tensor_slices(
        (
            {
                "numeric": X.astype(np.float32),
                "neighbourhood": neigh.astype(np.int32),
            },
            y.astype(np.float32),
        )
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 20000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def build_mlp_sequential(n_features: int, widths: list[int], dropout: float, batch_norm: bool) -> keras.Model:
    model = keras.Sequential(name="mlp_sequential")
    model.add(layers.Input(shape=(n_features,)))
    for i, width in enumerate(widths):
        model.add(layers.Dense(width, activation=None, name=f"dense_{i}"))
        if batch_norm:
            model.add(layers.BatchNormalization(name=f"bn_{i}"))
        model.add(layers.Activation("relu", name=f"relu_{i}"))
        if dropout and dropout > 0:
            model.add(layers.Dropout(dropout, seed=SEED, name=f"drop_{i}"))
    model.add(layers.Dense(1, activation="sigmoid", name="noshow_prob"))
    return model


def build_mlp_functional(n_features: int, n_neighbourhoods: int, embed_dim: int = 8) -> keras.Model:
    """Numeric branch plus a learned neighbourhood embedding."""
    numeric = layers.Input(shape=(n_features,), name="numeric")
    neigh = layers.Input(shape=(), dtype="int32", name="neighbourhood")
    emb = layers.Embedding(
        input_dim=n_neighbourhoods,
        output_dim=embed_dim,
        name="neighbourhood_embedding",
    )(neigh)
    emb = layers.Flatten(name="emb_flat")(emb)
    x = layers.Concatenate(name="join")([numeric, emb])
    x = layers.Dense(64, activation="relu", name="dense_0")(x)
    x = layers.Dropout(0.3, seed=SEED, name="drop_0")(x)
    x = layers.Dense(32, activation="relu", name="dense_1")(x)
    out = layers.Dense(1, activation="sigmoid", name="noshow_prob")(x)
    return keras.Model(inputs=[numeric, neigh], outputs=out, name="mlp_functional_embed")


def compile_binary(model: keras.Model, learning_rate: float) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model
