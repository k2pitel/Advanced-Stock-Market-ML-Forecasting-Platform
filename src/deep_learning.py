"""Deep learning model builders for MLP, LSTM, and GRU."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def build_mlp(input_dim: int):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_lstm(sequence_length: int, n_features: int):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(64, input_shape=(sequence_length, n_features), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_gru(sequence_length: int, n_features: int):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout

    model = Sequential([
        GRU(64, input_shape=(sequence_length, n_features), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def create_sequences(data: np.ndarray, sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)
