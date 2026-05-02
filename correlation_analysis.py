"""
Correlation Analysis: Teacher Mobility ↔ Student Performance
=============================================================
Loads a dataset containing teacher mobility scores (1–10) and student
performance scores (1–10), computes the Pearson correlation coefficient,
tests its statistical significance, and produces a labelled scatter plot
with a regression line.

Usage
-----
Run against the bundled sample data:
    python correlation_analysis.py

Run against a custom CSV file:
    python correlation_analysis.py my_data.csv

The CSV must contain columns named 'teacher_mobility' and
'student_performance'.

Outputs
-------
* Console summary: correlation coefficient, p-value, and interpretation.
* 'correlation_plot.png' – scatter plot with regression line saved in the
  current working directory.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend safe for CI / headless
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLOT_FILE = "correlation_plot.png"
SIGNIFICANCE_LEVEL = 0.05

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    """Load and validate the dataset from *csv_path*."""
    df = pd.read_csv(csv_path)
    required = {"teacher_mobility", "student_performance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = df[["teacher_mobility", "student_performance"]].dropna()
    if len(df) < 3:
        raise ValueError("Dataset must contain at least 3 rows for correlation analysis.")
    return df


def generate_synthetic_data() -> pd.DataFrame:
    """Return a small synthetic dataset as a fallback when no file is given."""
    rng = np.random.default_rng(42)
    mobility = rng.uniform(1, 10, size=60)
    noise = rng.normal(0, 0.5, size=60)
    performance = np.clip(10.0 - (mobility - 1.0) + noise, 1.0, 10.0)
    return pd.DataFrame(
        {"teacher_mobility": mobility, "student_performance": performance}
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_correlation(df: pd.DataFrame) -> dict:
    """
    Compute Pearson correlation between teacher_mobility and
    student_performance.  Returns a dict with:
        r       – correlation coefficient  (-1 to 1)
        p_value – two-tailed p-value
        n       – sample size
        slope   – slope of the OLS regression line
        intercept – intercept of the OLS regression line
    """
    x = df["teacher_mobility"].values
    y = df["student_performance"].values

    r, p_value = stats.pearsonr(x, y)
    slope, intercept, *_ = stats.linregress(x, y)

    return {
        "r": float(r),
        "r_squared": float(r ** 2),
        "p_value": float(p_value),
        "n": int(len(df)),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def interpret_correlation(result: dict) -> str:
    """Return a plain-English interpretation of the correlation result."""
    r = result["r"]
    p = result["p_value"]
    abs_r = abs(r)

    # Strength label
    if abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.4:
        strength = "moderate"
    elif abs_r >= 0.1:
        strength = "weak"
    else:
        strength = "negligible"

    direction = "negative" if r < 0 else "positive"
    significance = (
        f"statistically significant (p = {p:.4f} < {SIGNIFICANCE_LEVEL})"
        if p < SIGNIFICANCE_LEVEL
        else f"not statistically significant (p = {p:.4f} ≥ {SIGNIFICANCE_LEVEL})"
    )

    lines = [
        f"Pearson r = {r:.4f}  |  R² = {result['r_squared']:.4f}  |  p = {p:.4f}  |  n = {result['n']}",
        "",
        f"There is a {strength} {direction} correlation between teacher mobility",
        f"and student performance.  The relationship is {significance}.",
    ]

    if p < SIGNIFICANCE_LEVEL:
        if r < 0:
            lines.append(
                "\nInterpretation: Higher teacher mobility is associated with "
                "LOWER student performance."
            )
        else:
            lines.append(
                "\nInterpretation: Higher teacher mobility is associated with "
                "HIGHER student performance."
            )
    else:
        lines.append(
            "\nInterpretation: No reliable linear correlation was detected "
            "at the chosen significance level."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_correlation(
    df: pd.DataFrame,
    result: dict,
    output_path: str = PLOT_FILE,
) -> None:
    """
    Save a scatter plot of teacher_mobility vs student_performance with an
    OLS regression line and annotation of key statistics.
    """
    x = df["teacher_mobility"].values
    y = df["student_performance"].values

    x_line = np.linspace(x.min(), x.max(), 300)
    y_line = result["slope"] * x_line + result["intercept"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.6, color="steelblue", edgecolors="white",
               linewidth=0.5, zorder=3, label="Observations")
    ax.plot(x_line, y_line, color="crimson", linewidth=2,
            label=f"Regression line  (slope = {result['slope']:.3f})")

    # Annotation box
    annotation = (
        f"r = {result['r']:.4f}\n"
        f"R² = {result['r_squared']:.4f}\n"
        f"p = {result['p_value']:.4f}\n"
        f"n = {result['n']}"
    )
    ax.text(
        0.97, 0.97, annotation,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )

    ax.set_xlabel("Teacher Mobility (1–10)", fontsize=12)
    ax.set_ylabel("Student Performance (1–10)", fontsize=12)
    ax.set_title("Correlation: Teacher Mobility & Student Performance", fontsize=13)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Correlation plot saved → {output_path}")


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
        df = load_data(csv_path)
    elif os.path.isfile("sample_data.csv"):
        print("Loading data from 'sample_data.csv' …")
        df = load_data("sample_data.csv")
    else:
        print("No dataset provided – using synthetic data …")
        df = generate_synthetic_data()

    print(f"  {len(df)} samples loaded.\n")

    # Compute and display correlation
    result = compute_correlation(df)
    interpretation = interpret_correlation(result)

    print("=" * 60)
    print("CORRELATION ANALYSIS RESULTS")
    print("=" * 60)
    print(interpretation)
    print("=" * 60)

    # Plot
    print()
    plot_correlation(df, result)


if __name__ == "__main__":
    main()
