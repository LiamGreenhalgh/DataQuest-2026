"""
Neural Network: Teacher Mobility → Student Performance Predictor
================================================================
Trains a multi-layer perceptron (MLP) regression network on a dataset
of teacher mobility scores (1–10) and corresponding student performance
scores (1–10), then lets the user query a mobility value to receive a
predicted performance score.

Usage
-----
Train and save the model, then enter predictions interactively:
    python neural_network.py

To use a custom CSV dataset instead of the built-in sample data, pass the
file path as the first argument:
    python neural_network.py my_data.csv

The CSV must contain columns named 'teacher_mobility' and
'student_performance'.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend safe for CI / headless
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_FILE = "neural_network_model.pkl"
SCALER_FILE = "neural_network_scaler.pkl"
PLOT_FILE = "neural_network_training.png"

INPUT_MIN, INPUT_MAX = 1.0, 10.0
OUTPUT_MIN, OUTPUT_MAX = 1.0, 10.0

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load teacher_mobility and student_performance columns from *csv_path*."""
    df = pd.read_csv(csv_path)
    required = {"teacher_mobility", "student_performance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    X = df["teacher_mobility"].values.astype(float).reshape(-1, 1)
    y = df["student_performance"].values.astype(float)
    return X, y


def generate_synthetic_data(n_samples: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset that mimics a negative relationship between
    teacher mobility and student performance with added noise.
    Higher teacher turnover → lower student performance.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(INPUT_MIN, INPUT_MAX, size=(n_samples, 1))
    # linear base: performance = 10 - (mobility - 1) * (9/9) + noise
    noise = rng.normal(0, 0.4, size=n_samples)
    y = 10.0 - (X.squeeze() - 1.0) + noise
    y = np.clip(y, OUTPUT_MIN, OUTPUT_MAX)
    return X, y


# ---------------------------------------------------------------------------
# Model building / training
# ---------------------------------------------------------------------------

def build_and_train(X: np.ndarray, y: np.ndarray) -> tuple[MLPRegressor, MinMaxScaler]:
    """
    Scale inputs, build an MLPRegressor, train it, and return the fitted
    model together with the input scaler.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        max_iter=2000,
        learning_rate_init=0.001,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=50,
        verbose=False,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  Test MSE : {mse:.4f}")
    print(f"  Test R²  : {r2:.4f}")

    return model, scaler


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_predictions(
    model: MLPRegressor,
    scaler: MinMaxScaler,
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    output_path: str = PLOT_FILE,
) -> None:
    """Scatter the training data and overlay the neural network's predictions."""
    mobility_range = np.linspace(INPUT_MIN, INPUT_MAX, 300).reshape(-1, 1)
    mobility_scaled = scaler.transform(mobility_range)
    perf_pred = model.predict(mobility_scaled)
    perf_pred = np.clip(perf_pred, OUTPUT_MIN, OUTPUT_MAX)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X_raw, y_raw, alpha=0.5, color="steelblue", label="Training data", zorder=3)
    ax.plot(
        mobility_range,
        perf_pred,
        color="crimson",
        linewidth=2,
        label="Neural Network prediction",
    )
    ax.set_xlabel("Teacher Mobility (1–10)", fontsize=12)
    ax.set_ylabel("Student Performance (1–10)", fontsize=12)
    ax.set_title("Neural Network: Teacher Mobility → Student Performance", fontsize=13)
    ax.set_xlim(INPUT_MIN - 0.2, INPUT_MAX + 0.2)
    ax.set_ylim(OUTPUT_MIN - 0.5, OUTPUT_MAX + 0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Training plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: MLPRegressor, scaler: MinMaxScaler) -> None:
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Model saved → {MODEL_FILE}")
    print(f"  Scaler saved → {SCALER_FILE}")


def load_model() -> tuple[MLPRegressor, MinMaxScaler]:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def predict(mobility: float, model: MLPRegressor, scaler: MinMaxScaler) -> float:
    """Return predicted student performance for a given teacher mobility value."""
    if not (INPUT_MIN <= mobility <= INPUT_MAX):
        raise ValueError(f"Teacher mobility must be between {INPUT_MIN} and {INPUT_MAX}.")
    X = np.array([[mobility]])
    X_scaled = scaler.transform(X)
    performance = model.predict(X_scaled)[0]
    return float(np.clip(performance, OUTPUT_MIN, OUTPUT_MAX))


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def interactive_predict(model: MLPRegressor, scaler: MinMaxScaler) -> None:
    print("\n--- Prediction Mode ---")
    print("Enter a teacher mobility value (1–10) to get the predicted student")
    print("performance score, or type 'quit' to exit.\n")
    while True:
        raw = input("Teacher mobility (1–10): ").strip()
        if raw.lower() in {"quit", "exit", "q"}:
            break
        try:
            mobility = float(raw)
            perf = predict(mobility, model, scaler)
            print(f"  → Predicted student performance: {perf:.2f} / 10\n")
        except ValueError as exc:
            print(f"  Error: {exc}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Decide data source
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if not os.path.isfile(csv_path):
            print(f"Error: file not found – '{csv_path}'")
            sys.exit(1)
        print(f"Loading data from '{csv_path}' …")
        X, y = load_data(csv_path)
    elif os.path.isfile("sample_data.csv"):
        print("Loading data from 'sample_data.csv' …")
        X, y = load_data("sample_data.csv")
    else:
        print("No dataset provided – generating synthetic data …")
        X, y = generate_synthetic_data()

    print(f"  {len(X)} samples loaded.\n")

    # Train or load cached model
    if os.path.isfile(MODEL_FILE) and os.path.isfile(SCALER_FILE):
        answer = input("A saved model was found. Retrain? [y/n]: ").strip().lower()
        if answer == "y":
            print("\nTraining neural network …")
            model, scaler = build_and_train(X, y)
            save_model(model, scaler)
        else:
            model, scaler = load_model()
            print("  Loaded saved model.")
    else:
        print("Training neural network …")
        model, scaler = build_and_train(X, y)
        save_model(model, scaler)

    plot_predictions(model, scaler, X, y)
    interactive_predict(model, scaler)


if __name__ == "__main__":
    main()
