# sales-prediction-using-python
A Python project that explores advertising spend across TV, radio, and newspaper channels, then compares Linear Regression and Random Forest models to predict product sales. Includes a Jupyter notebook, visualizations, and model evaluation.
# Sales Prediction Using Python

This project predicts product sales from advertising spend across TV, radio, and newspaper channels. It includes a runnable Python script and a Jupyter notebook for exploration and model comparison.

## Project files

- `main.py` — loads the dataset, performs exploratory analysis, trains models, reports test metrics, and plots residuals.
- `sales_prediction.ipynb` — step-by-step notebook version of the analysis.
- `data/Advertising.csv` — the Advertising dataset used in the project.

## Requirements

Python 3.9 or later is recommended. Install the required packages with:

```bash
python -m pip install pandas scikit-learn matplotlib seaborn jupyter
```

## Run the Python script

From the project directory, run:

```bash
python main.py
```

The script loads `data/Advertising.csv`. If the file is absent, it downloads the dataset from the public [Advertising.csv source](https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv).

The script prints data checks and descriptive statistics, displays exploratory plots, trains Linear Regression and Random Forest models, compares MAE/RMSE/R² on a held-out test set, and plots residuals for the model with the lower test RMSE.

To open the notebook:

```bash
jupyter notebook sales_prediction.ipynb
```

## Dataset and features

The dataset contains 200 observations and four columns:

| Column | Description |
| --- | --- |
| `tv` | TV advertising spend |
| `radio` | Radio advertising spend |
| `newspaper` | Newspaper advertising spend |
| `sales` | Product sales target |

The analysis includes missing-value checks, descriptive statistics, a pairplot, individual sales-versus-channel scatter plots, and a correlation heatmap.

## Example test-set results

Using an 80/20 train/test split with `random_state=42`, the models produced:

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Random Forest | 0.662 | 0.791 | 0.980 |
| Linear Regression | 1.461 | 1.782 | 0.899 |

from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
DATA_PATH = Path(__file__).resolve().parent / "data" / "Advertising.csv"
DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
FEATURES = ["tv", "radio", "newspaper"]


def load_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the advertising CSV, downloading it if it is not present."""
    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urlretrieve(DATA_URL, data_path)
        except Exception as exc:
            raise RuntimeError(f"Could not download the dataset from {DATA_URL}") from exc

    data = pd.read_csv(data_path, index_col=0)
    data.columns = data.columns.str.strip().str.lower()

    expected_columns = FEATURES + ["sales"]
    if set(data.columns) != set(expected_columns):
        raise ValueError(
            f"Expected columns {expected_columns}; found {data.columns.tolist()}"
        )

    data = data[expected_columns]
    if data.isna().any().any():
        missing = data.isna().sum()
        raise ValueError(f"Dataset contains missing values:\n{missing[missing > 0]}")

    return data


def show_exploratory_plots(data: pd.DataFrame) -> None:
    """Display pairwise relationships, individual scatter plots, and correlations."""
    pair_plot = sns.pairplot(data, diag_kind="hist", corner=True)
    pair_plot.fig.suptitle("Advertising features and sales", y=1.02)
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for axis, channel in zip(axes, FEATURES):
        sns.scatterplot(data=data, x=channel, y="sales", ax=axis, alpha=0.75)
        axis.set_title(f"Sales vs. {channel.title()} spend")
        axis.set_xlabel(f"{channel.title()} spend")
        axis.set_ylabel("Sales")
    fig.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        data.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
    )
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    plt.show()


def main() -> None:
    data = load_data()
    print(f"Loaded {len(data)} rows from {DATA_PATH}")
    print("\nFirst five rows:")
    print(data.head())
    print("\nMissing values by column:")
    print(data.isna().sum())
    print("\nDescriptive statistics:")
    print(data.describe().T)

    show_exploratory_plots(data)

    X = data[FEATURES]
    y = data["sales"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            min_samples_leaf=2,
        ),
    }

    predictions = {}
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        results.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
                "R2": r2_score(y_test, y_pred),
            }
        )

    results_table = pd.DataFrame(results).set_index("Model").sort_values("RMSE")
    print("\nTest-set model comparison (lower MAE/RMSE, higher R2 are better):")
    print(results_table.round(3))

    best_model_name = results_table.index[0]
    best_predictions = predictions[best_model_name]
    print(f"\nBest model by test RMSE: {best_model_name}")

    residuals = y_test.to_numpy() - best_predictions
    fig, axis = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x=best_predictions, y=residuals, ax=axis, alpha=0.8)
    axis.axhline(0, color="red", linestyle="--", linewidth=1)
    axis.set(
        title=f"Residual plot: {best_model_name}",
        xlabel="Predicted sales",
        ylabel="Actual - predicted sales",
    )
    fig.tight_layout()
    plt.show()

    linear_coefficients = pd.Series(
        models["Linear Regression"].coef_,
        index=FEATURES,
        name="Linear regression coefficient",
    )
    forest_importances = pd.Series(
        models["Random Forest"].feature_importances_,
        index=FEATURES,
        name="Random forest importance",
    )
    interpretation = pd.concat(
        [linear_coefficients, forest_importances], axis=1
    ).sort_values("Random forest importance", ascending=False)
    print("\nChannel interpretation:")
    print(interpretation.round(4))

    highest_importance = forest_importances.idxmax()
    largest_coefficient = linear_coefficients.abs().idxmax()
    print(
        f"\nHighest random-forest feature importance: "
        f"{highest_importance.title()} "
        f"({forest_importances[highest_importance]:.3f})."
    )
    print(
        f"Largest linear coefficient by absolute value: {largest_coefficient.title()} "
        f"({linear_coefficients[largest_coefficient]:.3f} sales per spend unit)."
    )
    print("Feature importance and coefficients are predictive associations, not causal effects.")


if __name__ == "__main__":
    main()

On this split, TV had the highest Random Forest feature importance. The linear model's largest coefficient by absolute value was for radio. These are predictive associations from this dataset, not causal estimates or guarantees of future performance.
